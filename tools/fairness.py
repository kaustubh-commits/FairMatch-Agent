"""
Fairness metrics for a set of assignments (output of tools.allocation.allocate).

  avg_fit                mean skill-fit score of every filled slot (0-1, higher = better matches)
  gini                   inequality of TOTAL opportunities per student
                         (past + newly assigned), over all students.
                         0 = perfectly equal, 1 = one person has everything.
  unique_recipients      how many different students got at least one slot
  repeat_selection_rate  share of filled slots that went to students who
                         already had >= 1 past opportunity
  avg_commute_km         mean home-to-listing distance of filled slots (km)
  outskirts_gap_pts      selection rate of everyone else MINUS the selection
                         rate of "outskirts" students (home > OUTSKIRTS_KM from
                         MG Road), in percentage points. Positive = outskirts
                         students are picked less often. A commute penalty can
                         cause this, so the Equity Analyst flags gaps above
                         OUTSKIRTS_FLAG_PTS.
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.constants import BANGALORE_AREAS, distance_km
from data_pipeline.schemas import Student
from tools.store import load_students

CITY_CENTRE = BANGALORE_AREAS["MG Road"]
OUTSKIRTS_KM = 12.0
OUTSKIRTS_FLAG_PTS = 10.0


def is_outskirts(student: Student) -> bool:
    """True if the student's home is more than OUTSKIRTS_KM from MG Road."""
    return distance_km((student.lat, student.lon), CITY_CENTRE) > OUTSKIRTS_KM


def gini(values: list[float]) -> float:
    """
    Gini coefficient of a list of non-negative numbers.

    Formula on values sorted ascending x_1..x_n:
        G = 2 * sum(i * x_i) / (n * sum(x)) - (n + 1) / n
    """
    values = sorted(values)
    n, total = len(values), sum(values)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum(i * x for i, x in enumerate(values, start=1))
    return round(2 * weighted / (n * total) - (n + 1) / n, 4)


def fairness_metrics(assignments: list[dict],
                     students: Optional[list[Student]] = None) -> dict:
    """Compute the fairness metrics (plus slots_filled) for one allocation."""
    students = students if students is not None else load_students()
    past = {s.id: s.past_opportunities for s in students}

    new_counts = {sid: 0 for sid in past}
    fits, commutes = [], []
    for a in assignments:
        for sid in a["student_ids"]:
            new_counts[sid] += 1
            # Skill fit is comparable across policies; fall back to the score if absent.
            fits.append(a.get("skill_fits", a["scores"])[sid])
            km = a.get("commutes", {}).get(sid)
            if km is not None:
                commutes.append(km)

    slots_filled = len(fits)
    repeat_slots = sum(1 for a in assignments for sid in a["student_ids"] if past[sid] >= 1)

    # Outskirts selection gap: compare the share of each group selected at least once.
    outskirts = {s.id for s in students if is_outskirts(s)}
    selected = {sid for sid, c in new_counts.items() if c > 0}
    others = set(past) - outskirts

    def rate(group: set) -> float:
        return len(group & selected) / len(group) if group else 0.0

    outskirts_rate, others_rate = rate(outskirts), rate(others)
    return {
        "avg_fit": round(sum(fits) / slots_filled, 3) if slots_filled else 0.0,
        "gini": gini([past[sid] + new_counts[sid] for sid in past]),
        "unique_recipients": len(selected),
        "repeat_selection_rate": round(repeat_slots / slots_filled, 3) if slots_filled else 0.0,
        "avg_commute_km": round(sum(commutes) / len(commutes), 1) if commutes else None,
        "outskirts_rate": round(outskirts_rate, 3),
        "others_rate": round(others_rate, 3),
        "outskirts_gap_pts": round((others_rate - outskirts_rate) * 100, 1),
        "slots_filled": slots_filled,
    }
