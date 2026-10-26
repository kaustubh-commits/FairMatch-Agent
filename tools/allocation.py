"""
Allocate ALL opportunities to students under one of three policies.

  skill_first   pick the best match score, full stop (commute weight 0.2)
  equal_access  among eligible students, pick those with the FEWEST past
                opportunities first; skill fit only breaks ties (commute ignored)
  hybrid        priority = match score x access adjustment, where the adjustment
                1 / (1 + ACCESS_PENALTY * past_opportunities) gently favours
                students who have had fewer chances (commute weight 0.1)

The commute weight per policy lives in tools/scoring.py (POLICY_COMMUTE_WEIGHT).

Shared rules for every policy (so the comparison is fair):
  - only eligible students (see tools/scoring.py), optionally within `max_km`
  - each opportunity fills at most `slots` students
  - a student can win at most MAX_PER_STUDENT opportunities in one run
  - opportunities are processed in id order (simple greedy algorithm)
  - ties go to the student with fewer past opportunities, then lower id
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.schemas import Opportunity, Student
from tools.scoring import POLICY_COMMUTE_WEIGHT, score_match
from tools.store import load_opportunities, load_students

POLICIES = ("skill_first", "equal_access", "hybrid")
MAX_PER_STUDENT = 2
ACCESS_PENALTY = 0.5


# Plain-English statement of each priority rule below (used in explanations).
POLICY_RULES = {
    "skill_first": "candidates are ranked by match score (80% skill fit, 20% closeness to home); "
                   "past opportunities play no role",
    "equal_access": "eligible candidates with the fewest past opportunities go first; "
                    "skill fit only breaks ties; the commute is ignored",
    "hybrid": f"candidates are ranked by match score (90% skill fit, 10% closeness to home) divided by "
              f"(1 + {ACCESS_PENALTY} x past opportunities), so a long history lowers priority",
}


def priority(policy: str, score: float, past: int) -> tuple:
    """Sort key for a candidate under a policy. Bigger = picked earlier."""
    if policy == "skill_first":
        return (score,)
    if policy == "equal_access":
        return (-past, score)          # fewer past opportunities first
    if policy == "hybrid":
        return (round(score / (1 + ACCESS_PENALTY * past), 6),)
    raise ValueError(f"Unknown policy '{policy}'. Choose one of {POLICIES}.")


def rank_for_policy(opportunity: Opportunity, policy: str,
                    students: list[Student], max_km: Optional[float] = None) -> list[dict]:
    """
    All eligible students for one opportunity, ordered by the policy.
    Each item is score_match(...) plus `past_opportunities` and `priority`.
    """
    weight = POLICY_COMMUTE_WEIGHT[policy] if policy in POLICY_COMMUTE_WEIGHT else priority(policy, 0, 0)
    candidates = []
    for student in students:
        match = score_match(student, opportunity, weight, max_km)
        if match["eligible"]:
            match["past_opportunities"] = student.past_opportunities
            match["priority"] = priority(policy, match["score"], student.past_opportunities)
            candidates.append(match)
    # Tie-break first (fewer past opportunities, then id), then sort by
    # priority. Python's sort is stable, so equal priorities keep the
    # tie-break order: a tie never favours a student who already had chances.
    candidates.sort(key=lambda m: (m["past_opportunities"], m["student_id"]))
    candidates.sort(key=lambda m: m["priority"], reverse=True)
    return candidates


def allocate(policy: str, students: Optional[list[Student]] = None,
             opportunities: Optional[list[Opportunity]] = None,
             max_km: Optional[float] = None) -> list[dict]:
    """
    Assign students to every opportunity under `policy`.

    Returns one dict per opportunity:
      {opportunity_id, policy, student_ids, scores{sid: match score},
       skill_fits{sid: skill fit}, commutes{sid: km or None}, runner_up}
    Skill fit is reported separately because match scores of different
    policies use different commute weights and are not directly comparable.
    """
    priority(policy, 0.0, 0)  # fail fast on an unknown policy
    students = students if students is not None else load_students()
    opportunities = opportunities if opportunities is not None else load_opportunities()

    wins = {s.id: 0 for s in students}   # how many opportunities each student has won
    assignments = []
    for opp in sorted(opportunities, key=lambda o: o.id):
        ranked = [m for m in rank_for_policy(opp, policy, students, max_km)
                  if wins[m["student_id"]] < MAX_PER_STUDENT]
        chosen = ranked[:opp.slots]
        for m in chosen:
            wins[m["student_id"]] += 1
        assignments.append({
            "opportunity_id": opp.id,
            "policy": policy,
            "student_ids": [m["student_id"] for m in chosen],
            "scores": {m["student_id"]: m["score"] for m in chosen},
            "skill_fits": {m["student_id"]: m["skill_fit"] for m in chosen},
            "commutes": {m["student_id"]: m["commute_km"] for m in chosen},
            "runner_up": ranked[opp.slots]["student_id"] if len(ranked) > opp.slots else None,
        })
    return assignments
