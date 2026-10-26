"""
Equity Analyst: checks whether a proposal concentrates opportunities on
students who already had them, and compares policies.

All numbers come from tools.fairness.fairness_metrics. The LLM (if any)
may only rephrase the policy takeaway, and its sentence is rejected unless
every number in it is grounded in the metrics (see agents/grounding.py).
"""

from __future__ import annotations

import json

from agents.context import RunContext
from agents.grounding import ungrounded
from tools.decisions import make_decision
from tools.fairness import fairness_metrics
from tools.store import load_students

NAME = "Equity Analyst"
FAIR_ALTERNATIVE = "hybrid"


def _baseline_share() -> float:
    """Share of all students who already had at least one opportunity."""
    students = load_students()
    return round(sum(1 for s in students if s.past_opportunities >= 1) / len(students), 3)


def check_proposals(match: dict, ctx: RunContext) -> dict:
    """
    Flag a proposal if a bigger share of its picks already had opportunities
    than in the student population as a whole. If flagged, show what the
    fairer `hybrid` policy would pick instead (without logging it).
    """
    proposals = match["proposals"]
    assignments = [{"opportunity_id": r.opportunity_id, "student_ids": r.selected,
                    "scores": {sid: r.score_breakdown[sid]["score"] for sid in r.selected}}
                   for r in proposals.values()]

    with ctx.trace.step(NAME, "check concentration", tool="fairness_metrics",
                        policy=match["policy"], opportunities=list(proposals)) as ev:
        metrics = fairness_metrics(assignments)
        baseline = _baseline_share()
        flagged = metrics["slots_filled"] > 0 and metrics["repeat_selection_rate"] > baseline
        ev.result_summary = (f"repeat-selection {metrics['repeat_selection_rate']:.0%} "
                             f"vs {baseline:.0%} of all students with prior opportunities"
                             + (" -> FLAG" if flagged else " -> OK"))

    alternatives = {}
    if flagged and match["policy"] != FAIR_ALTERNATIVE:
        for opp_id in proposals:
            with ctx.trace.step(NAME, f"alternative for {opp_id}", tool="make_decision",
                                opportunity_id=opp_id, policy=FAIR_ALTERNATIVE) as ev:
                alt = make_decision(opp_id, FAIR_ALTERNATIVE)
                alternatives[opp_id] = alt
                ev.result_summary = f"{FAIR_ALTERNATIVE} would select {alt.selected}"

    if flagged:
        message = (f"Equity flag: {metrics['repeat_selection_rate']:.0%} of the proposed picks "
                   f"already had an opportunity, versus {baseline:.0%} of all students. "
                   f"This proposal would increase concentration.")
        if alternatives:
            message += "".join(f" For {o} the {FAIR_ALTERNATIVE} policy would select "
                               f"{', '.join(a.selected) or 'nobody'}."
                               for o, a in alternatives.items())
    else:
        message = (f"Equity check passed: {metrics['repeat_selection_rate']:.0%} of picks "
                   f"had prior opportunities (population: {baseline:.0%}).")
    ctx.trace.note(NAME, "verdict", message)
    return {"metrics": metrics, "baseline": baseline, "flagged": flagged,
            "alternatives": alternatives, "message": message}


LABEL = {"skill_first": "skill-first", "equal_access": "equal access", "hybrid": "hybrid"}


def template_takeaway(table: dict) -> str:
    """One-line takeaway built only from the metrics table."""
    sf, ea, hy = table["skill_first"], table["equal_access"], table["hybrid"]
    return (f"Hybrid keeps avg fit at {hy['avg_fit']:.3f} (skill-first {sf['avg_fit']:.3f}) "
            f"while cutting repeat selection from {sf['repeat_selection_rate']:.1%} to "
            f"{hy['repeat_selection_rate']:.1%} and Gini from {sf['gini']:.3f} to {hy['gini']:.3f}; "
            f"equal access is the most equal (Gini {ea['gini']:.3f}) but has the lowest fit "
            f"({ea['avg_fit']:.3f}).")


def template_rounds_takeaway(sims: dict) -> str:
    """One-line takeaway for the multi-round simulation, built only from its results."""
    sf, ea, hy = sims["skill_first"][-1], sims["equal_access"][-1], sims["hybrid"][-1]
    n = sf["round"]
    return (f"After {n} rounds, skill-first still gives {sf['repeat_selection_rate']:.0%} of slots to "
            f"the previous round's winners and reaches only {sf['cumulative_reach']} students "
            f"(Gini {sf['gini']:.3f}), while hybrid reaches {hy['cumulative_reach']} "
            f"(Gini {hy['gini']:.3f}) and equal access {ea['cumulative_reach']} (Gini {ea['gini']:.3f}).")


def _sim_facts(sims: dict) -> dict:
    """The simulation numbers the LLM may quote (kept small)."""
    keys = ("round", "gini", "repeat_selection_rate", "cumulative_reach", "avg_fit")
    return {p: [{k: r[k] for k in keys} for r in rounds] for p, rounds in sims.items()}


TAKEAWAY_PROMPT = """You are the Equity Analyst of an internship allocation system.
Below are results for three allocation policies, computed by deterministic tools.

"snapshot": one allocation of every opportunity.
  avg_fit: mean skill fit (higher = better matches). gini: inequality of total opportunities
  per student (lower = fairer). repeat_selection_rate: share of slots going to students who
  already had an opportunity. unique_recipients: distinct students selected.
"rounds": the same allocation repeated; after each round winners gain +1 past opportunity.
  repeat_selection_rate here = share of the round's slots won by the PREVIOUS round's winners.
  cumulative_reach = students selected at least once so far.

Return JSON {{"snapshot": "...", "rounds": "..."}}: one sentence each (max 45 words) for a
placement officer, on the trade-off and on what repetition does. Use only numbers that appear
below (you may round them or write rates as percentages). Do not invent any other number.

{facts}"""


def compare_policies(allocations: dict, sims: dict, ctx: RunContext) -> dict:
    """Metrics for each policy, plus one-line takeaways for the snapshot and the rounds."""
    table = {}
    for policy, assignments in allocations.items():
        with ctx.trace.step(NAME, f"metrics ({policy})", tool="fairness_metrics", policy=policy) as ev:
            table[policy] = fairness_metrics(assignments)
            m = table[policy]
            ev.result_summary = (f"avg fit {m['avg_fit']:.3f}, gini {m['gini']:.3f}, "
                                 f"unique {m['unique_recipients']}, "
                                 f"repeat {m['repeat_selection_rate']:.1%}")

    takeaways = {"snapshot": template_takeaway(table), "rounds": template_rounds_takeaway(sims)}
    sources = {"snapshot": "template", "rounds": "template"}
    facts = {"snapshot": table, "rounds": _sim_facts(sims)}
    # One LLM call for both sentences (the free tier allows few calls per day).
    reply = ctx.ask_llm(NAME, "write takeaways",
                        TAKEAWAY_PROMPT.format(facts=json.dumps(facts, indent=1)), json_mode=True)
    if isinstance(reply, dict):
        for key in ("snapshot", "rounds"):
            text = reply.get(key)
            if not isinstance(text, str) or not text.strip():
                continue
            problems = ungrounded(text, facts[key])
            if problems:
                ctx.trace.note(NAME, f"LLM {key} takeaway rejected",
                               f"ungrounded values {problems[:5]}; using template.")
            else:
                takeaways[key], sources[key] = " ".join(text.split()), "llm"
    for key in ("snapshot", "rounds"):
        ctx.trace.note(NAME, f"{key} takeaway ({sources[key]})", takeaways[key])
    return {"table": table, "takeaway": takeaways["snapshot"], "takeaway_source": sources["snapshot"],
            "rounds_takeaway": takeaways["rounds"], "rounds_takeaway_source": sources["rounds"]}
