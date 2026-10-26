"""
Equity Analyst: checks whether a proposal concentrates opportunities on
students who already had them, and compares policies.

It also watches the OUTSKIRTS SELECTION GAP: a commute penalty makes
internships more practical but can push out students who live far from the
centre, so a policy whose gap exceeds OUTSKIRTS_FLAG_PTS is flagged.

All numbers come from tools.fairness.fairness_metrics. The LLM (if any)
may only rephrase the policy takeaway, and its sentence is rejected unless
every number in it is grounded in the metrics (see agents/grounding.py).
"""

from __future__ import annotations

import json

from agents.context import RunContext
from agents.grounding import ungrounded
from agents.trace import shorten
from tools.decisions import make_decision
from tools.fairness import OUTSKIRTS_FLAG_PTS, OUTSKIRTS_KM, fairness_metrics
from tools.store import get_student, load_students

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
                    "scores": {sid: r.score_breakdown[sid]["score"] for sid in r.selected},
                    "skill_fits": {sid: r.score_breakdown[sid].get("skill_fit") for sid in r.selected},
                    "commutes": {sid: r.score_breakdown[sid].get("commute_km") for sid in r.selected}}
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
                alt = make_decision(opp_id, FAIR_ALTERNATIVE,
                                    max_km=proposals[opp_id].constraints.get("max_km"))
                alternatives[opp_id] = alt
                ev.result_summary = f"{FAIR_ALTERNATIVE} would select {', '.join(alt.selected) or 'nobody'}"

    def names(ids: list[str]) -> str:
        return ", ".join(get_student(sid).label for sid in ids) or "nobody"

    if flagged:
        message = (f"Equity flag: {metrics['repeat_selection_rate']:.0%} of the proposed picks "
                   f"already had an opportunity, versus {baseline:.0%} of all students. "
                   f"This proposal would increase concentration.")
        if alternatives:
            message += "".join(f" For {o} the {FAIR_ALTERNATIVE} policy would select {names(a.selected)}."
                               for o, a in alternatives.items())
        alt_ids = [sid for a in alternatives.values() for sid in a.selected]
        headline = (f"Flagged: {metrics['repeat_selection_rate']:.0%} of the picks already had an "
                    f"opportunity, versus {baseline:.0%} of all students"
                    + (f"; the hybrid policy would pick {names(list(dict.fromkeys(alt_ids)))} instead."
                       if alt_ids else "."))
    else:
        message = (f"Equity check passed: {metrics['repeat_selection_rate']:.0%} of picks "
                   f"had prior opportunities (population: {baseline:.0%}).")
        headline = (f"No concentration concern: {metrics['repeat_selection_rate']:.0%} of the picks "
                    f"had prior opportunities, close to the {baseline:.0%} of all students.")
    ctx.trace.note(NAME, "verdict", message)
    ctx.trace.headline(NAME, headline)
    return {"metrics": metrics, "baseline": baseline, "flagged": flagged,
            "alternatives": alternatives, "message": message}


LABEL = {"skill_first": "skill-first", "equal_access": "equal access", "hybrid": "hybrid"}


def template_takeaway(table: dict) -> str:
    """One-line takeaway built only from the metrics table."""
    sf, ea, hy = table["skill_first"], table["equal_access"], table["hybrid"]
    return (f"Hybrid keeps the average skill fit at {hy['avg_fit']:.3f} (skill-first {sf['avg_fit']:.3f}) "
            f"while cutting repeat selection from {sf['repeat_selection_rate']:.1%} to "
            f"{hy['repeat_selection_rate']:.1%} and Gini from {sf['gini']:.3f} to {hy['gini']:.3f}; "
            f"equal access is the most equal (Gini {ea['gini']:.3f}) but has the lowest fit "
            f"({ea['avg_fit']:.3f}).")


def outskirts_flags(table: dict) -> list[str]:
    """Policies whose outskirts selection gap exceeds the flag threshold."""
    return [p for p, m in table.items() if m["outskirts_gap_pts"] > OUTSKIRTS_FLAG_PTS]


def outskirts_sentence(table: dict) -> str:
    flagged = outskirts_flags(table)
    if not flagged:
        worst = max(table, key=lambda p: table[p]["outskirts_gap_pts"])
        return (f"Students living more than {OUTSKIRTS_KM:g} km from MG Road are picked about as often as "
                f"others under every policy (largest gap {table[worst]['outskirts_gap_pts']:.1f} points, "
                f"{LABEL[worst]}).")
    parts = ", ".join(f"{LABEL[p]} {table[p]['outskirts_gap_pts']:.1f} points" for p in flagged)
    return (f"Outskirts flag: students living more than {OUTSKIRTS_KM:g} km from MG Road are picked less "
            f"often ({parts}; threshold {OUTSKIRTS_FLAG_PTS:g}).")


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
    keys = ("round", "gini", "repeat_selection_rate", "cumulative_reach", "avg_fit", "outskirts_gap_pts")
    return {p: [{k: r[k] for k in keys} for r in rounds] for p, rounds in sims.items()}


TAKEAWAY_PROMPT = """You are the Equity Analyst of an internship allocation system.
Below are results for three allocation policies, computed by deterministic tools.

"snapshot": one allocation of every opportunity.
  avg_fit: mean skill fit (higher = better matches). gini: inequality of total opportunities
  per student (lower = fairer). repeat_selection_rate: share of slots going to students who
  already had an opportunity. unique_recipients: distinct students selected.
  avg_commute_km: average home-to-listing distance. outskirts_gap_pts: how many percentage
  points less often students living far from the centre are selected (above 10 is a concern).
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
                                 f"repeat {m['repeat_selection_rate']:.1%}, "
                                 f"commute {m['avg_commute_km']} km, outskirts gap {m['outskirts_gap_pts']} pts")

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
    # Only note template text: accepted LLM text is already in the LLM step above.
    for key in ("snapshot", "rounds"):
        if sources[key] == "template":
            ctx.trace.note(NAME, f"{key} takeaway (template)", takeaways[key])

    outskirts = outskirts_sentence(table)
    flagged = outskirts_flags(table)
    ctx.trace.note(NAME, "outskirts check", outskirts)
    ctx.trace.headline(NAME, shorten(takeaways["snapshot"], 220)
                       + (f" Outskirts flag for {', '.join(LABEL[p] for p in flagged)}." if flagged else ""))
    return {"table": table, "takeaway": takeaways["snapshot"], "takeaway_source": sources["snapshot"],
            "rounds_takeaway": takeaways["rounds"], "rounds_takeaway_source": sources["rounds"],
            "outskirts": outskirts, "outskirts_flags": flagged}
