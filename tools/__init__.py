"""
Deterministic tools (PLAN.md section 4).

Every score, ranking, allocation and reason in FairMatch comes from these
plain functions. The agents in Phase 2 call them; the LLM never invents one.

Most tools accept optional `students` / `opportunities` lists. If you leave
them out, the processed data in data/processed/ is used. Tests pass small
hand-made lists instead.
"""

from tools.allocation import POLICIES, allocate
from tools.decisions import (appeal_decision, approve_decision, get_decision,
                             list_decisions, log_decision, make_decision)
from tools.fairness import fairness_metrics, gini
from tools.recommend import recommend_for_profile, skills_to_learn
from tools.scoring import rank_candidates, score_match
from tools.search import find_opportunities_near, search_opportunities, search_students
from tools.simulation import simulate_rounds

__all__ = [
    "POLICIES", "allocate", "appeal_decision", "approve_decision", "get_decision",
    "list_decisions", "log_decision", "make_decision", "fairness_metrics", "gini", "rank_candidates",
    "score_match", "find_opportunities_near", "search_opportunities", "search_students",
    "recommend_for_profile", "skills_to_learn", "simulate_rounds",
]
