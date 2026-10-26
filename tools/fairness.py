"""
Fairness metrics for a set of assignments (output of tools.allocation.allocate).

  avg_fit                mean skill-fit score of every filled slot (0-1, higher = better matches)
  gini                   inequality of TOTAL opportunities per student
                         (past + newly assigned), over all students.
                         0 = perfectly equal, 1 = one person has everything.
  unique_recipients      how many different students got at least one slot
  repeat_selection_rate  share of filled slots that went to students who
                         already had >= 1 past opportunity
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.schemas import Student
from tools.store import load_students


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
    """Compute the four PLAN.md metrics (plus slots_filled) for one allocation."""
    students = students if students is not None else load_students()
    past = {s.id: s.past_opportunities for s in students}

    new_counts = {sid: 0 for sid in past}
    fits = []
    for a in assignments:
        for sid in a["student_ids"]:
            new_counts[sid] += 1
            fits.append(a["scores"][sid])

    slots_filled = len(fits)
    repeat_slots = sum(1 for a in assignments for sid in a["student_ids"] if past[sid] >= 1)

    return {
        "avg_fit": round(sum(fits) / slots_filled, 3) if slots_filled else 0.0,
        "gini": gini([past[sid] + new_counts[sid] for sid in past]),
        "unique_recipients": sum(1 for c in new_counts.values() if c > 0),
        "repeat_selection_rate": round(repeat_slots / slots_filled, 3) if slots_filled else 0.0,
        "slots_filled": slots_filled,
    }
