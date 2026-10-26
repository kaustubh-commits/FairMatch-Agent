"""
Multi-round simulation: what happens to inequality if a policy is used
again and again?

Each round allocates every opportunity under the policy. Every student
selected in a round gains +1 past opportunity before the next round, so a
policy that keeps choosing the same students makes them look even more
"experienced" next time: the rich-get-richer loop the Policy Lab shows.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Optional

from data_pipeline.schemas import Opportunity, Student
from tools.allocation import allocate
from tools.fairness import fairness_metrics
from tools.store import load_opportunities, load_students

DEFAULT_ROUNDS = 4


def simulate_rounds(policy: str, rounds: int = DEFAULT_ROUNDS,
                    students: Optional[list[Student]] = None,
                    opportunities: Optional[list[Opportunity]] = None,
                    max_km: Optional[float] = None) -> list[dict]:
    """
    Run `rounds` allocation rounds under `policy` (optionally with a commute limit).

    Returns one dict per round: fairness_metrics(...) for that round, plus
      round                  1, 2, ...
      repeat_selection_rate  REDEFINED per round: share of this round's slots won
                             by students who were selected in the PREVIOUS round
                             (for round 1: students with any real past opportunity).
                             "Ever had one" would hit 100% for every policy after a
                             round or two, because 132 slots go to 150 students;
                             "won last time too" shows who keeps the chances.
      cumulative_reach       students selected at least once in rounds 1..n
    The original student list is never modified (we work on copies).
    """
    pool = list(students if students is not None else load_students())
    opportunities = opportunities if opportunities is not None else load_opportunities()

    reached: set[str] = set()
    previous = {s.id for s in pool if s.past_opportunities >= 1}   # "round 0" = real history
    history = []
    for r in range(1, rounds + 1):
        assignments = allocate(policy, pool, opportunities, max_km)
        metrics = fairness_metrics(assignments, pool)
        wins = Counter(sid for a in assignments for sid in a["student_ids"])
        slots = sum(wins.values())
        repeat = sum(n for sid, n in wins.items() if sid in previous) / slots if slots else 0.0
        reached |= set(wins)
        history.append({"round": r, **metrics, "repeat_selection_rate": round(repeat, 3),
                        "cumulative_reach": len(reached)})
        previous = set(wins)
        # Winners carry their new opportunity into the next round.
        pool = [replace(s, past_opportunities=s.past_opportunities + wins[s.id]) for s in pool]
    return history
