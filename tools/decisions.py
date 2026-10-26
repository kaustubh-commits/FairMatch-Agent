"""
Decision log: create, store, approve and appeal allocation decisions.

Decisions are kept in a JSON file (data/processed/decisions.json by
default; tests pass their own `path`). No database, as PLAN.md requires.

Appeal rule (deterministic, run by the Auditor in Phase 2):
  The appellant (a given student, or the runner-up by default) is
  re-scored with the same policy. The decision is REVISED if either
    a) a selected student is no longer eligible, or
    b) the appellant's fit is within APPEAL_MARGIN of the weakest selected
       student AND the appellant has had fewer past opportunities
  in which case the appellant replaces that student. Otherwise it is UPHELD.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from data_pipeline.build import PROCESSED_DIR
from data_pipeline.schemas import Appeal, DecisionRecord, Opportunity, Student
from tools.allocation import MAX_PER_STUDENT, rank_for_policy
from tools.scoring import MIN_HOURS, POLICY_COMMUTE_WEIGHT, score_match
from tools.store import get_opportunity, get_student, load_opportunities, load_students

DEFAULT_PATH = PROCESSED_DIR / "decisions.json"
APPEAL_MARGIN = 0.05


# ------------------------------------------------------------- storage -----

def _load_all(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _save_all(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))


def _update(record: DecisionRecord, path: Path) -> None:
    """Replace the stored record that has the same id."""
    records = [record.to_dict() if r["id"] == record.id else r for r in _load_all(path)]
    _save_all(records, path)


# Keys in a record's score_breakdown[sid] that are not skills.
BREAKDOWN_FIELDS = ("score", "skill_fit", "commute_km")


def _breakdown(match: dict) -> dict:
    """Compact breakdown for the record: match score, skill fit, commute, and {skill: points}."""
    return {"score": match["score"], "skill_fit": match["skill_fit"], "commute_km": match["commute_km"],
            **{skill: b["points"] for skill, b in match["breakdown"].items()}}


def _past(n: int) -> str:
    return f"{n} past opportunit{'y' if n == 1 else 'ies'}"


def _summary(match: dict) -> str:
    """'match 0.86 (skill fit 0.92, commute 4.9 km)' from a score_match result."""
    commute = f", commute {match['commute_km']} km" if match["commute_km"] is not None else ""
    return f"match {match['score']:.2f} (skill fit {match['skill_fit']:.2f}{commute})"


def _reason_line(match: dict, label: str) -> str:
    """One readable sentence built only from tool outputs."""
    skills = ", ".join(f"{s} {b['level']}/3" for s, b in match["breakdown"].items())
    return (f"{match['student_label']} {label}: {_summary(match)}; {skills}; "
            f"{_past(match['past_opportunities'])}.")


# ------------------------------------------------------------- tools -------

def make_decision(opportunity_id: str, policy: str = "hybrid",
                  students: Optional[list[Student]] = None,
                  opportunities: Optional[list[Opportunity]] = None,
                  prior_wins: Optional[dict[str, int]] = None,
                  max_km: Optional[float] = None) -> DecisionRecord:
    """
    Build (but don't save) a DecisionRecord for one opportunity: the top
    `slots` candidates under `policy`, plus the runner-up and the reasons.

    `prior_wins` counts how many opportunities each student has already been
    selected for in the same request. Students at MAX_PER_STUDENT are skipped,
    the next candidate takes their place, and the skip is written into the
    record (reasons + constraints["skipped"]) so nobody disappears silently.
    `max_km` (optional) makes students with a longer commute ineligible.
    """
    students = students if students is not None else load_students()
    opp = get_opportunity(opportunity_id,
                          opportunities if opportunities is not None else load_opportunities())
    prior_wins = prior_wins or {}
    ranked = rank_for_policy(opp, policy, students, max_km)

    available, skipped = [], []
    for m in ranked:
        if prior_wins.get(m["student_id"], 0) >= MAX_PER_STUDENT:
            # Only worth reporting if they would otherwise have been picked or runner-up.
            if len(available) <= opp.slots:
                skipped.append(m)
        else:
            available.append(m)
    selected = available[:opp.slots]
    runner_up = available[opp.slots] if len(available) > opp.slots else None

    reasons = [_reason_line(m, "selected") for m in selected]
    if runner_up:
        reasons.append(_reason_line(runner_up, "runner-up"))
    skip_reasons = {
        m["student_id"]: (f"already selected for {prior_wins[m['student_id']]} other opportunities "
                          f"in this request (max {MAX_PER_STUDENT} per student)")
        for m in skipped}
    reasons += [f"{m['student_label']} skipped: {_summary(m)}, but "
                f"{skip_reasons[m['student_id']]}." for m in skipped]
    if not selected:
        reasons.append("No eligible candidates for this opportunity."
                       if not ranked else "Every eligible candidate is at the per-student cap.")

    shown = selected + ([runner_up] if runner_up else []) + skipped
    constraints = {"slots": opp.slots, "min_hours": MIN_HOURS,
                   "max_per_student": MAX_PER_STUDENT, "eligible_candidates": len(ranked),
                   "commute_weight": POLICY_COMMUTE_WEIGHT.get(policy, 0.0)}
    if max_km is not None:
        constraints["max_km"] = max_km
    if skip_reasons:
        constraints["skipped"] = skip_reasons
    return DecisionRecord(
        opportunity_id=opp.id,
        selected=[m["student_id"] for m in selected],
        runner_up=runner_up["student_id"] if runner_up else None,
        policy=policy,
        score_breakdown={m["student_id"]: _breakdown(m) for m in shown},
        reasons=reasons,
        constraints=constraints,
    )


def log_decision(record: DecisionRecord, path: Path = DEFAULT_PATH) -> DecisionRecord:
    """Save a new decision. Assigns an id (D0001, ...) and a UTC timestamp."""
    records = _load_all(path)
    record.id = f"D{len(records) + 1:04d}"
    record.timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record.status = "proposed"
    records.append(record.to_dict())
    _save_all(records, path)
    return record


def get_decision(decision_id: str, path: Path = DEFAULT_PATH) -> DecisionRecord:
    """Fetch one decision by id (raises KeyError if missing)."""
    for r in _load_all(path):
        if r["id"] == decision_id:
            return DecisionRecord.from_dict(r)
    raise KeyError(f"Unknown decision id: {decision_id}")


def list_decisions(path: Path = DEFAULT_PATH) -> list[DecisionRecord]:
    """All logged decisions, oldest first."""
    return [DecisionRecord.from_dict(r) for r in _load_all(path)]


def approve_decision(decision_id: str, path: Path = DEFAULT_PATH) -> DecisionRecord:
    """A human approves the decision (status -> approved)."""
    record = get_decision(decision_id, path)
    record.status = "approved"
    _update(record, path)
    return record


def appeal_decision(decision_id: str, reason: str, student_id: Optional[str] = None,
                    path: Path = DEFAULT_PATH,
                    students: Optional[list[Student]] = None,
                    opportunities: Optional[list[Opportunity]] = None) -> DecisionRecord:
    """
    Re-evaluate a decision after an appeal (see the rule at the top of this file).
    Status becomes "revised" if the outcome changes, otherwise "appealed".
    """
    students = students if students is not None else load_students()
    record = get_decision(decision_id, path)
    opp = get_opportunity(record.opportunity_id,
                          opportunities if opportunities is not None else load_opportunities())
    appellant_id = student_id or record.runner_up

    weight = record.constraints.get("commute_weight", POLICY_COMMUTE_WEIGHT.get(record.policy, 0.0))
    max_km = record.constraints.get("max_km")

    def rescore(sid: str) -> dict:
        student = get_student(sid, students)
        match = score_match(student, opp, weight, max_km)
        match["past_opportunities"] = student.past_opportunities
        return match

    def who(m: dict) -> str:
        return f"{m['student_label']} (match {m['score']:.2f}, {_past(m['past_opportunities'])})"

    outcome, explanation = "upheld", ""
    if appellant_id is None:
        explanation = "No appellant or runner-up to compare against; decision upheld."
    elif appellant_id in record.selected:
        explanation = f"{appellant_id} is already selected; nothing to change."
    else:
        appellant = rescore(appellant_id)
        selected = [rescore(sid) for sid in record.selected]
        ineligible = [m for m in selected if not m["eligible"]]
        # The "weakest" selected student = lowest fit (ties: more past opportunities).
        weakest = min(selected, key=lambda m: (m["score"], -m["past_opportunities"]), default=None)

        if not appellant["eligible"]:
            explanation = (f"{appellant['student_label']} is not eligible: "
                           + " ".join(r.removeprefix("Not eligible: ") for r in appellant["reasons"]
                                      if r.startswith("Not eligible")))
        elif ineligible:
            replaced = ineligible[0]
            outcome = "revised"
            explanation = (f"{replaced['student_label']} no longer meets eligibility; "
                           f"replaced by eligible appellant {who(appellant)}.")
        elif weakest is None:
            replaced = None
            outcome = "revised"
            explanation = f"No one was selected; {appellant_id} is eligible and is now selected."
        elif (weakest["score"] - appellant["score"] <= APPEAL_MARGIN
              and appellant["past_opportunities"] < weakest["past_opportunities"]):
            replaced = weakest
            outcome = "revised"
            explanation = (f"{who(appellant)} is within {APPEAL_MARGIN} of {who(weakest)} "
                           f"and has had fewer opportunities; selection revised.")
        else:
            gap = weakest["score"] - appellant["score"]
            if gap <= APPEAL_MARGIN:   # close enough on score, so history decided it
                explanation = (f"{who(appellant)} is within {APPEAL_MARGIN} of the weakest selected "
                               f"student, {who(weakest)}, but a revision also needs fewer past "
                               f"opportunities. Decision upheld.")
            else:
                explanation = (f"{who(appellant)} scores {gap:.2f} below the weakest selected student, "
                               f"{who(weakest)}; a revision needs a gap of at most {APPEAL_MARGIN} and fewer "
                               f"past opportunities. Decision upheld.")

        if outcome == "revised":
            if replaced is not None:
                record.selected = [appellant_id if sid == replaced["student_id"] else sid
                                   for sid in record.selected]
                # The displaced student is now the runner-up (the appellant can't be both).
                record.runner_up = replaced["student_id"]
                record.score_breakdown.setdefault(replaced["student_id"], _breakdown(replaced))
            else:
                record.selected = [appellant_id]
            record.score_breakdown[appellant_id] = _breakdown(appellant)

    record.status = "revised" if outcome == "revised" else "appealed"
    record.appeal = Appeal(text=reason, student_id=appellant_id,
                           outcome=outcome, explanation=explanation)
    record.reasons.append(f"Appeal ({outcome}): {explanation}")
    _update(record, path)
    return record
