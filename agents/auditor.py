"""
Auditor: verifies proposals against the rules, logs them, and handles
explain/appeal lookups. No LLM: rules are rules.

Rules checked (same constants as the tools):
  - every selected student is eligible (skills + MIN_HOURS)
  - no opportunity has more students than slots
  - no student is selected more than MAX_PER_STUDENT times in one run
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from agents.context import RunContext
from data_pipeline.schemas import DecisionRecord
from tools.allocation import MAX_PER_STUDENT
from tools.decisions import appeal_decision, get_decision, list_decisions, log_decision
from tools.scoring import score_match
from tools.store import get_opportunity, get_student

NAME = "Auditor"


def verify_and_log(proposals: dict[str, DecisionRecord], path: Path, ctx: RunContext) -> list[DecisionRecord]:
    """Check each proposal, drop rule-breaking picks, then log it as 'proposed'."""
    logged = []
    wins = Counter()
    for opp_id, record in proposals.items():
        opp = get_opportunity(opp_id)
        with ctx.trace.step(NAME, f"verify {opp_id}", tool="score_match",
                            selected=record.selected, slots=opp.slots) as ev:
            problems = []
            for sid in list(record.selected):
                match = score_match(get_student(sid), opp)
                if not match["eligible"]:
                    problems.append(f"{sid} ineligible")
                    record.selected.remove(sid)
                elif wins[sid] >= MAX_PER_STUDENT:
                    problems.append(f"{sid} over the {MAX_PER_STUDENT}-per-student cap")
                    record.selected.remove(sid)
                else:
                    wins[sid] += 1
            if len(record.selected) > opp.slots:
                problems.append(f"{len(record.selected)} selected for {opp.slots} slots")
                record.selected = record.selected[:opp.slots]
            ev.result_summary = ("OK: eligible, within slots and per-student cap" if not problems
                                 else "FIXED: " + "; ".join(problems))

        with ctx.trace.step(NAME, f"log {opp_id}", tool="log_decision", opportunity_id=opp_id) as ev:
            saved = log_decision(record, path)
            logged.append(saved)
            ev.result_summary = f"{saved.id} logged (status {saved.status}, awaiting human approval)"
    return logged


def check_allocations(allocations: dict, ctx: RunContext) -> dict:
    """Confirm every policy's allocation respects slots and the per-student cap."""
    report = {}
    for policy, assignments in allocations.items():
        with ctx.trace.step(NAME, f"audit ({policy})", tool="check allocation rules",
                            policy=policy) as ev:
            wins = Counter(sid for a in assignments for sid in a["student_ids"])
            over_cap = [sid for sid, n in wins.items() if n > MAX_PER_STUDENT]
            over_slots = [a["opportunity_id"] for a in assignments
                          if len(a["student_ids"]) > get_opportunity(a["opportunity_id"]).slots]
            report[policy] = {"over_cap": over_cap, "over_slots": over_slots}
            ev.result_summary = ("OK" if not (over_cap or over_slots)
                                 else f"VIOLATIONS cap={over_cap} slots={over_slots}")
    return report


def _resolve_id(decision_id: Optional[str], path: Path, ctx: RunContext) -> Optional[str]:
    """Use the given id, or the most recent decision if none was named."""
    if decision_id:
        return decision_id
    with ctx.trace.step(NAME, "find latest decision", tool="list_decisions") as ev:
        records = list_decisions(path)
        ev.result_summary = f"latest is {records[-1].id}" if records else "no decisions logged yet"
    return records[-1].id if records else None


def fetch(decision_id: Optional[str], path: Path, ctx: RunContext) -> Optional[DecisionRecord]:
    decision_id = _resolve_id(decision_id, path, ctx)
    if decision_id is None:
        return None
    with ctx.trace.step(NAME, "fetch decision", tool="get_decision", decision_id=decision_id) as ev:
        try:
            record = get_decision(decision_id, path)
            ev.result_summary = f"{record.id}: {record.opportunity_id}, selected {record.selected}, {record.status}"
            return record
        except KeyError:
            ev.result_summary = f"{decision_id} not found"
            return None


def appeal(decision_id: Optional[str], reason: str, student_id: Optional[str],
           path: Path, ctx: RunContext) -> Optional[DecisionRecord]:
    decision_id = _resolve_id(decision_id, path, ctx)
    if decision_id is None:
        return None
    with ctx.trace.step(NAME, "re-evaluate appeal", tool="appeal_decision", decision_id=decision_id,
                        student_id=student_id or "(runner-up)", reason=reason) as ev:
        try:
            record = appeal_decision(decision_id, reason, student_id=student_id, path=path)
            ev.result_summary = f"{record.appeal.outcome}: {record.appeal.explanation}"[:200]
            return record
        except KeyError as e:
            ev.result_summary = f"failed: {e}"
            return None
