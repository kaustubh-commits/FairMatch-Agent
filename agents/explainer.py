"""
Explainer: writes explanations ONLY from DecisionRecord fields (or, for a
student, only from the recommendation tools' results).

The detailed explanation is always a template filled from the record. In
LLM mode the LLM also writes a short plain-language summary. To keep the
app fast and within free-tier quotas, ONE LLM call covers every decision in
a request. Each summary is shown only if every number and id in it appears
in that decision's facts (agents/grounding.py); otherwise that decision
falls back to its template.
"""

from __future__ import annotations

import json

from agents.context import RunContext
from agents.grounding import ungrounded
from agents.trace import shorten
from data_pipeline.schemas import DecisionRecord
from tools.allocation import POLICY_RULES
from tools.store import get_student

NAME = "Explainer"
POLICY_NAMES = {"skill_first": "skill-first", "equal_access": "equal-access", "hybrid": "hybrid"}

SUMMARY_PROMPT = """You explain internship allocation decisions to students and placement officers.
Here are the facts of {n} decision(s), produced by deterministic tools:

{facts}

For EACH decision write 2-3 short, plain sentences: who was selected, why, who is the
runner-up, anyone skipped and why, and (if present) the appeal outcome. Refer to students
by their label, e.g. "Ananya R. (S025)".
Rules:
- The "policy_rule" is the ONLY reason anyone was selected. Never say someone was
  chosen because of something the policy rule does not use.
- "past opportunities" are listed for transparency; mention them only as the policy rule allows.
- Use only facts above. Do not add any number, student id or reason that is not there.
Return JSON: {{"<decision_id>": "<summary>", ...}} with one entry per decision."""


def _label(sid: str) -> str:
    try:
        return get_student(sid).label
    except KeyError:
        return sid


def record_facts(record: DecisionRecord) -> dict:
    """
    The facts the LLM may use, all taken from the record: who was picked, the
    record's own reasons, and the plain-English rule of the policy it used.
    (The raw per-skill points are left out; the reasons already state them.)
    """
    facts = {
        "decision_id": record.id,
        "opportunity_id": record.opportunity_id,
        "policy": record.policy,
        "policy_rule": POLICY_RULES.get(record.policy, ""),
        "selected": [_label(sid) for sid in record.selected],
        "runner_up": _label(record.runner_up) if record.runner_up else None,
        "skipped": {_label(s): why for s, why in record.constraints.get("skipped", {}).items()},
        "reasons": record.reasons,
        "status": record.status,
    }
    if record.constraints.get("max_km"):
        facts["commute_limit_km"] = record.constraints["max_km"]
    if record.appeal:
        facts["appeal"] = {"student": _label(record.appeal.student_id), "text": record.appeal.text,
                           "outcome": record.appeal.outcome}
    return facts


def template_summary(record: DecisionRecord) -> str:
    def with_match(sid: str) -> str:
        b = record.score_breakdown.get(sid, {})
        if b.get("score") is None:
            return _label(sid)
        commute = f", {b['commute_km']} km away" if b.get("commute_km") is not None else ""
        return f"{_label(sid)} [match {b['score']:.2f}{commute}]"

    if record.selected:
        picks = ", ".join(with_match(sid) for sid in record.selected)
        verb = "was" if len(record.selected) == 1 else "were"
        text = f"Under the {POLICY_NAMES.get(record.policy, record.policy)} policy, {picks} {verb} selected"
    else:
        text = f"Under the {POLICY_NAMES.get(record.policy, record.policy)} policy, nobody was selected"
    if record.runner_up:
        text += f"; {with_match(record.runner_up)} is the runner-up"
    text += "."
    for sid, why in record.constraints.get("skipped", {}).items():
        text += f" {_label(sid)} was skipped: {why}."
    if record.appeal:
        text += f" Appeal by {_label(record.appeal.student_id)}: {record.appeal.outcome}."
    return text


def details_text(record: DecisionRecord, summary: str) -> str:
    """The full, template-only explanation (reasons straight from the record)."""
    lines = [f"Decision {record.id} | opportunity {record.opportunity_id} | "
             f"policy {record.policy} | status {record.status}"]
    lines += [f"  - {reason}" for reason in record.reasons]
    c = record.constraints
    if c:
        lines.append("  Rules applied: " + ", ".join(f"{k}={v}" for k, v in c.items() if k != "skipped"))
    if record.appeal:  # the appeal's explanation is already one of the reasons above
        lines.append(f"  Appeal by {_label(record.appeal.student_id)} (\"{record.appeal.text}\"): "
                     f"{record.appeal.outcome}.")
    return f"Summary: {summary}\n" + "\n".join(lines)


def explain_all(records: list[DecisionRecord], ctx: RunContext) -> tuple[dict, dict]:
    """
    Explain several decisions with ONE LLM call.
    Returns ({decision_id: summary}, {decision_id: full explanation text}).
    """
    summaries = {r.id: template_summary(r) for r in records}
    facts = {r.id: record_facts(r) for r in records}
    accepted, rejected = [], []

    if records:
        reply = ctx.ask_llm(NAME, f"explain {len(records)} decision{'' if len(records) == 1 else 's'}",
                            SUMMARY_PROMPT.format(n=len(records), facts=json.dumps(facts, indent=1)),
                            json_mode=True)
        if isinstance(reply, dict):
            for r in records:
                text = reply.get(r.id)
                if not isinstance(text, str) or not text.strip():
                    rejected.append(f"{r.id}: missing")
                    continue
                problems = ungrounded(text, facts[r.id])
                if problems:
                    rejected.append(f"{r.id}: ungrounded {problems[:5]}")
                else:
                    summaries[r.id] = " ".join(text.split())
                    accepted.append(r.id)
        if rejected:
            ctx.trace.note(NAME, "LLM text rejected", "; ".join(rejected) + " -> template used.")
        if not accepted:   # template only: say so once (accepted LLM text is already in the LLM step)
            ctx.trace.note(NAME, "explanation (template)",
                           shorten(" ".join(summaries.values()), 400))

    texts = {r.id: details_text(r, summaries[r.id]) for r in records}
    if accepted and not rejected:
        how = "Gemini wrote it and every number was checked against the records"
    elif accepted:
        how = "Gemini wrote part of it; the rest uses the rule-based template"
    elif ctx.llm is None:
        how = "rule-based template"
    else:
        how = "Gemini's text failed the number check, so the rule-based template was used"
    noun = "record" if len(records) == 1 else "records"
    ctx.trace.headline(NAME, f"Wrote the explanation from the decision {noun} ({how}).")
    return summaries, texts


def explain(record: DecisionRecord, ctx: RunContext) -> tuple[str, str]:
    """Returns (summary, full explanation text) for one decision."""
    summaries, texts = explain_all([record], ctx)
    return summaries[record.id], texts[record.id]


# ------------------------------------------------------ recommendations ----

MATCHES_PROMPT = """You help a student understand which internships fit them.
Here are results produced by deterministic tools (match is 0-1 and combines skill fit with
closeness to home; "learn_next" lists missing skills and how many extra listings in the
student's interests each would make them eligible for; "also_opens" counts listings in other domains):

{facts}

Write 2-3 short, encouraging, plain sentences: the best match and why (which skills),
how many listings they qualify for, and the most useful skill to learn next.
Use ONLY facts above. Do not add any number, listing or skill that is not there. Plain text only."""


def match_facts(result: dict) -> dict:
    learn = result["learn"]
    return {
        "eligible_count": result["eligible_count"],
        "hours_per_week": result["hours"],
        "interests": learn["interests"],
        "matches": [{"opportunity_id": m["opportunity_id"], "title": m["title"], "company": m["company"],
                     "match": round(m["score"], 2), "skill_fit": round(m["skill_fit"], 2),
                     "commute_km": m["commute_km"],
                     "skills": {s: f"{b['level']}/3 ({b['kind']})" for s, b in m["breakdown"].items()}}
                    for m in result["matches"]],
        "learn_next": [{"skill": r["skill"], "unlocks_in_interests": r["unlocks_in_interests"],
                        "also_opens": r["unlocks_elsewhere"]} for r in learn["next"]],
    }


def template_matches(result: dict) -> str:
    if not result["hours_ok"]:
        return (f"With {result['hours']} hours a week you are below the minimum for these listings; "
                f"most need at least 8 hours.")
    matches, learn = result["matches"], result["learn"]
    if not matches:
        text = "No listing fits this profile yet."
    else:
        best = matches[0]
        strong = [s for s, b in best["breakdown"].items() if b["level"] >= 2]
        commute = f", {best['commute_km']} km from home" if best["commute_km"] is not None else ""
        text = (f"Your best match is {best['title']} at {best['company']} (match {best['score']:.2f}"
                + (f", thanks to {', '.join(strong)}" if strong else "") + f"{commute}). "
                f"You qualify for {result['eligible_count']} listing"
                f"{'' if result['eligible_count'] == 1 else 's'} in total.")
    if learn["next"]:
        top = learn["next"][0]
        n = top["unlocks_in_interests"]
        text += (f" Learning {top['skill']} would open {n} more listing{'' if n == 1 else 's'} "
                 f"in your interests ({', '.join(learn['interests'])}).")
    elif learn["also_opens"]:
        top = learn["also_opens"][0]
        text += (f" Nothing new opens in your interests yet; {top['skill']} would open "
                 f"{top['unlocks_elsewhere']} listings in other areas.")
    return text


def explain_matches(result: dict, ctx: RunContext) -> str:
    """A short summary of a student's matches, written only from the tool results (one LLM call)."""
    summary, source = template_matches(result), "template"
    facts = match_facts(result)
    if result["matches"] or result["learn"]["next"]:
        text = ctx.ask_llm(NAME, "summarise matches", MATCHES_PROMPT.format(facts=json.dumps(facts, indent=1)))
        if text:
            problems = ungrounded(text, facts)
            if problems:
                ctx.trace.note(NAME, "LLM text rejected", f"ungrounded values {problems[:5]}; using template.")
            else:
                summary, source = " ".join(text.split()), "llm"
    if source == "template":
        ctx.trace.note(NAME, "summary (template)", shorten(summary, 400))
    how = ("Gemini wrote it and every number was checked" if source == "llm" else "rule-based template")
    ctx.trace.headline(NAME, f"Summarised the matches in plain words ({how}).")
    return summary
