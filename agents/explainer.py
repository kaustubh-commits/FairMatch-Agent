"""
Explainer: writes the explanation ONLY from DecisionRecord fields.

The detailed explanation is always a template filled from the record. In
LLM mode the LLM additionally writes a short plain-language summary of the
same record; it is shown only if every number and id in it appears in the
record (agents/grounding.py). Otherwise the template summary is used.
"""

from __future__ import annotations

import json

from agents.context import RunContext
from agents.grounding import ungrounded
from data_pipeline.schemas import DecisionRecord
from tools.allocation import POLICY_RULES

NAME = "Explainer"

SUMMARY_PROMPT = """You explain internship allocation decisions to students and placement officers.
Here are the facts of one decision, produced by deterministic tools:

{facts}

Write 2-3 short, plain sentences: who was selected, why, who is the runner-up, and
(if present) the appeal outcome.
Rules:
- The "policy_rule" is the ONLY reason anyone was selected. Never say someone was
  chosen because of something the policy rule does not use.
- "past opportunities" are listed for transparency; mention them only as the policy rule allows.
- Use only facts above. Do not add any number, student id or reason that is not there.
Plain text only."""


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
        "selected": record.selected,
        "runner_up": record.runner_up,
        "reasons": record.reasons,
        "status": record.status,
    }
    if record.appeal:
        facts["appeal"] = {"student_id": record.appeal.student_id, "text": record.appeal.text,
                           "outcome": record.appeal.outcome}
    return facts


def template_summary(record: DecisionRecord) -> str:
    def with_fit(sid: str) -> str:
        score = record.score_breakdown.get(sid, {}).get("score")
        return f"{sid} (fit {score:.2f})" if score is not None else sid

    if record.selected:
        picks = ", ".join(with_fit(sid) for sid in record.selected)
        verb = "was" if len(record.selected) == 1 else "were"
        text = f"Under the {record.policy} policy, {picks} {verb} selected"
    else:
        text = f"Under the {record.policy} policy, nobody was selected"
    if record.runner_up:
        text += f"; {with_fit(record.runner_up)} is the runner-up"
    text += "."
    skipped = record.constraints.get("skipped", {})
    for sid, why in skipped.items():
        text += f" {sid} was skipped: {why}."
    if record.appeal:
        text += f" Appeal by {record.appeal.student_id}: {record.appeal.outcome}."
    return text


def explain(record: DecisionRecord, ctx: RunContext) -> tuple[str, str]:
    """Returns (summary, full explanation text) for one decision."""
    with ctx.trace.step(NAME, f"explain {record.id}", tool="DecisionRecord",
                        decision_id=record.id) as ev:
        lines = [f"Decision {record.id} | opportunity {record.opportunity_id} | "
                 f"policy {record.policy} | status {record.status}"]
        lines += [f"  - {reason}" for reason in record.reasons]
        c = record.constraints
        if c:
            lines.append("  Rules applied: " + ", ".join(f"{k}={v}" for k, v in c.items() if k != "skipped"))
        if record.appeal:  # the appeal's explanation is already one of the reasons above
            lines.append(f"  Appeal by {record.appeal.student_id} (\"{record.appeal.text}\"): "
                         f"{record.appeal.outcome}.")
        ev.result_summary = f"{len(record.reasons)} reasons from the record"

    summary, source = template_summary(record), "template"
    facts = record_facts(record)
    text = ctx.ask_llm(NAME, f"summarise {record.id}",
                       SUMMARY_PROMPT.format(facts=json.dumps(facts, indent=1)))
    if text:
        problems = ungrounded(text, facts)
        if problems:
            ctx.trace.note(NAME, "LLM summary rejected",
                           f"ungrounded values {problems[:5]}; using template.")
        else:
            summary, source = " ".join(text.split()), "llm"
    ctx.trace.note(NAME, f"summary ({source})", summary[:200])
    return summary, f"Summary: {summary}\n" + "\n".join(lines)


# ------------------------------------------------------ recommendations ----

MATCHES_PROMPT = """You help a student understand which internships fit them.
Here are results produced by deterministic tools (fit is 0-1; "learn" lists missing skills
and how many extra listings each would make the student eligible for):

{facts}

Write 2-3 short, encouraging, plain sentences: the best match and why (which skills),
how many listings they qualify for, and the most useful skill to learn next.
Use ONLY facts above. Do not add any number, listing or skill that is not there. Plain text only."""


def match_facts(result: dict) -> dict:
    return {
        "eligible_count": result["eligible_count"],
        "hours_per_week": result["hours"],
        "matches": [{"opportunity_id": m["opportunity_id"], "title": m["title"], "company": m["company"],
                     "fit": round(m["score"], 2),
                     "skills": {s: f"{b['level']}/3 ({b['kind']})" for s, b in m["breakdown"].items()}}
                    for m in result["matches"]],
        "learn": [{"skill": r["skill"], "unlocks": r["unlocks"]} for r in result["learn"]],
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
        text = (f"Your best match is {best['title']} at {best['company']} (fit {best['score']:.2f}"
                + (f", thanks to {', '.join(strong)}" if strong else "") + "). "
                f"You qualify for {result['eligible_count']} listing"
                f"{'' if result['eligible_count'] == 1 else 's'} in total.")
    if learn:
        top = learn[0]
        text += (f" Learning {top['skill']} would open {top['unlocks']} more listing"
                 f"{'' if top['unlocks'] == 1 else 's'}.")
    return text


def explain_matches(result: dict, ctx: RunContext) -> str:
    """A short summary of a student's matches, written only from the tool results."""
    summary, source = template_matches(result), "template"
    facts = match_facts(result)
    if result["matches"] or result["learn"]:
        text = ctx.ask_llm(NAME, "summarise matches", MATCHES_PROMPT.format(facts=json.dumps(facts, indent=1)))
        if text:
            problems = ungrounded(text, facts)
            if problems:
                ctx.trace.note(NAME, "LLM summary rejected",
                               f"ungrounded values {problems[:5]}; using template.")
            else:
                summary, source = " ".join(text.split()), "llm"
    ctx.trace.note(NAME, f"summary ({source})", summary[:200])
    return summary
