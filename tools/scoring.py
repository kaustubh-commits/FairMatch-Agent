"""
Scoring: how well does a student fit an opportunity?

The score is transparent on purpose. It has two parts:

1. SKILL FIT (0-1)
   - each REQUIRED skill is worth 2 points, each PREFERRED skill 1 point
   - a student earns (level / 3) of those points, e.g. level 2 on a required
     skill = 2/3 * 2 = 1.33 points
   - skill fit = points earned / points possible

2. CLOSENESS (0-1), from the commute
   - commute_km = straight-line (haversine) distance from the student's home
     to the opportunity's area
   - closeness = 1 - commute_km / REFERENCE_KM (never below 0)

   match score = (1 - w) * skill fit + w * closeness

The commute weight `w` is a POLICY choice (POLICY_COMMUTE_WEIGHT below).
Short commutes make an internship more practical, but penalising distance
pushes out students who live on the outskirts, so each policy sets its own
weight and the Equity Analyst measures the "outskirts selection gap".
When a student's home is unknown, the match score is just the skill fit.

Eligibility (hard rules, checked separately from the score):
  - covers at least half the required skills at level >= 1
  - has at least MIN_HOURS hours per week available
  - (optional) commute no longer than `max_km`
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.constants import BANGALORE_AREAS, distance_km
from data_pipeline.schemas import Opportunity, Student
from tools.store import get_opportunity, load_opportunities, load_students

REQUIRED_WEIGHT = 2
PREFERRED_WEIGHT = 1
MAX_LEVEL = 3
MIN_HOURS = 8
MIN_REQUIRED_COVERAGE = 0.5

REFERENCE_KM = 25.0          # roughly the width of the city: 25 km away = closeness 0
DEFAULT_COMMUTE_WEIGHT = 0.2
# How much each policy cares about the commute (see the module docstring).
POLICY_COMMUTE_WEIGHT = {"skill_first": 0.2, "hybrid": 0.1, "equal_access": 0.0}


def commute_km(student: Student, opportunity: Opportunity) -> Optional[float]:
    """Home-to-listing distance in km, or None if the student's home area is unknown."""
    if student.home_area not in BANGALORE_AREAS:
        return None
    return round(distance_km((student.lat, student.lon), (opportunity.lat, opportunity.lon)), 1)


def score_match(student: Student, opportunity: Opportunity,
                commute_weight: float = DEFAULT_COMMUTE_WEIGHT,
                max_km: Optional[float] = None) -> dict:
    """
    Score one student against one opportunity.

    Returns {student_id, student_label, opportunity_id, score, skill_fit, commute_km,
    closeness, commute_weight, breakdown, eligible, reasons}, where
    breakdown[skill] = {level, kind, weight, points, max_points}.
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
    skill_fit = round(earned / possible, 3) if possible else 0.0

    km = commute_km(student, opportunity)
    if km is None:
        closeness, score = None, skill_fit
    else:
        closeness = round(max(0.0, 1 - km / REFERENCE_KM), 3)
        score = round((1 - commute_weight) * skill_fit + commute_weight * closeness, 3)

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
    if max_km is not None and km is not None and km > max_km:
        eligible = False
        reasons.append(f"Not eligible: commute {km} km exceeds the {max_km:g} km limit.")

    # Human-readable lines, straight from the breakdown.
    for skill, b in breakdown.items():
        reasons.append(f"{skill} ({b['kind']}): level {b['level']}/{MAX_LEVEL} "
                       f"-> {b['points']:.2f}/{b['max_points']} pts")
    if km is not None:
        reasons.append(f"Commute: {km} km -> closeness {closeness:.2f} (weight {commute_weight:g})")

    return {
        "student_id": student.id,
        "student_label": student.label,
        "opportunity_id": opportunity.id,
        "score": score,
        "skill_fit": skill_fit,
        "commute_km": km,
        "closeness": closeness,
        "commute_weight": commute_weight,
        "breakdown": breakdown,
        "eligible": eligible,
        "reasons": reasons,
    }


def rank_candidates(opportunity_id: str, k: int = 3,
                    students: Optional[list[Student]] = None,
                    opportunities: Optional[list[Opportunity]] = None,
                    policy: str = "skill_first",
                    max_km: Optional[float] = None) -> list[dict]:
    """
    Top-k ELIGIBLE students for an opportunity, best match first, using the
    commute weight of `policy`. Ties are broken by FEWER past opportunities,
    then student id, so equal scores never favour students who already had chances.
    Each result is score_match(...) plus `past_opportunities`.
    """
    students = students if students is not None else load_students()
    opportunity = get_opportunity(opportunity_id,
                                  opportunities if opportunities is not None else load_opportunities())
    weight = POLICY_COMMUTE_WEIGHT.get(policy, DEFAULT_COMMUTE_WEIGHT)

    results = []
    for student in students:
        match = score_match(student, opportunity, weight, max_km)
        if match["eligible"]:
            match["past_opportunities"] = student.past_opportunities
            results.append(match)

    results.sort(key=lambda m: (-m["score"], m["past_opportunities"], m["student_id"]))
    return results[:k]
