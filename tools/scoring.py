"""
Scoring: how well does a student fit an opportunity?

The score is transparent on purpose:
  - each REQUIRED skill is worth 2 points, each PREFERRED skill 1 point
  - a student earns (level / 3) of those points, e.g. level 2 on a required
    skill = 2/3 * 2 = 1.33 points
  - score = points earned / points possible, so it is always between 0 and 1

Eligibility (a hard rule, checked separately from the score):
  - covers at least half the required skills at level >= 1, and
  - has at least MIN_HOURS hours per week available
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.schemas import Opportunity, Student
from tools.store import get_opportunity, load_opportunities, load_students

REQUIRED_WEIGHT = 2
PREFERRED_WEIGHT = 1
MAX_LEVEL = 3
MIN_HOURS = 8
MIN_REQUIRED_COVERAGE = 0.5


def score_match(student: Student, opportunity: Opportunity) -> dict:
    """
    Score one student against one opportunity.

    Returns {student_id, opportunity_id, score, breakdown, eligible, reasons}
    where breakdown[skill] = {level, kind, weight, points, max_points}.
    """
    breakdown = {}
    skill_weights = [(s, "required", REQUIRED_WEIGHT) for s in opportunity.required_skills]
    skill_weights += [(s, "preferred", PREFERRED_WEIGHT) for s in opportunity.preferred_skills]

    for skill, kind, weight in skill_weights:
        level = student.skills.get(skill, 0)
        breakdown[skill] = {
            "level": level,
            "kind": kind,
            "weight": weight,
            "points": round(level / MAX_LEVEL * weight, 3),
            "max_points": weight,
        }

    earned = sum(b["points"] for b in breakdown.values())
    possible = sum(b["max_points"] for b in breakdown.values())
    score = round(earned / possible, 3) if possible else 0.0

    # --- eligibility rules ---
    reasons = []
    required = opportunity.required_skills
    covered = [s for s in required if student.skills.get(s, 0) >= 1]
    coverage = len(covered) / len(required) if required else 1.0
    eligible = True
    if coverage < MIN_REQUIRED_COVERAGE:
        eligible = False
        reasons.append(f"Not eligible: has {len(covered)}/{len(required)} required skills "
                       f"(needs at least {MIN_REQUIRED_COVERAGE:.0%}).")
    if student.hours_per_week < MIN_HOURS:
        eligible = False
        reasons.append(f"Not eligible: {student.hours_per_week} hours/week available "
                       f"(minimum {MIN_HOURS}).")

    # Human-readable skill lines, straight from the breakdown.
    for skill, b in breakdown.items():
        reasons.append(f"{skill} ({b['kind']}): level {b['level']}/{MAX_LEVEL} "
                       f"-> {b['points']:.2f}/{b['max_points']} pts")

    return {
        "student_id": student.id,
        "opportunity_id": opportunity.id,
        "score": score,
        "breakdown": breakdown,
        "eligible": eligible,
        "reasons": reasons,
    }


def rank_candidates(opportunity_id: str, k: int = 3,
                    students: Optional[list[Student]] = None,
                    opportunities: Optional[list[Opportunity]] = None) -> list[dict]:
    """
    Top-k ELIGIBLE students for an opportunity, best fit first.
    Ties are broken by FEWER past opportunities, then student id, so equal
    scores never favour students who already had chances.
    Each result is score_match(...) plus `past_opportunities`.
    """
    students = students if students is not None else load_students()
    opportunity = get_opportunity(opportunity_id,
                                  opportunities if opportunities is not None else load_opportunities())

    results = []
    for student in students:
        match = score_match(student, opportunity)
        if match["eligible"]:
            match["past_opportunities"] = student.past_opportunities
            results.append(match)

    results.sort(key=lambda m: (-m["score"], m["past_opportunities"], m["student_id"]))
    return results[:k]
