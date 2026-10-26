"""
Matcher: runs the scoring / ranking / allocation tools. No LLM.

  rank intent    -> find opportunities, rank candidates, draft a DecisionRecord
  compare intent -> allocate all opportunities under each policy
"""

from __future__ import annotations

from agents.context import RunContext
from agents.planner import Plan
from tools.allocation import POLICIES, allocate
from tools.decisions import make_decision
from tools.scoring import rank_candidates
from tools.search import find_opportunities_near, search_opportunities

NAME = "Matcher"
MAX_OPPORTUNITIES = 3        # rank at most this many opportunities per request
DEFAULT_POLICY = "skill_first"   # "best candidates" = best fit; Equity Analyst checks fairness


def find_opportunities(plan: Plan, ctx: RunContext) -> dict:
    """Opportunities for the request, with the nearest-area fallback."""
    if plan.domain:
        with ctx.trace.step(NAME, "find opportunities", tool="find_opportunities_near",
                            domain=plan.domain, area=plan.area or "") as ev:
            if plan.area:
                found = find_opportunities_near(plan.domain, plan.area)
            else:
                opps = search_opportunities({"domain": plan.domain})
                found = {"opportunities": opps, "fell_back": False, "note": "",
                         "area_used": None, "requested_area": None}
            ev.result_summary = (f"{len(found['opportunities'])} found"
                                 + (f" in {found['area_used']}" if found["area_used"] else "")
                                 + (f". {found['note']}" if found["note"] else ""))
    else:
        filters = {"area": plan.area} if plan.area else {}
        with ctx.trace.step(NAME, "find opportunities", tool="search_opportunities", **filters) as ev:
            opps = search_opportunities(filters)
            found = {"opportunities": opps, "fell_back": False, "area_used": plan.area,
                     "requested_area": plan.area,
                     "note": "" if opps else f"No opportunities found for {filters or 'the request'}."}
            ev.result_summary = f"{len(opps)} found"

    if len(found["opportunities"]) > MAX_OPPORTUNITIES:
        ctx.trace.note(NAME, "limit", f"Using the first {MAX_OPPORTUNITIES} of "
                                      f"{len(found['opportunities'])} opportunities.")
        found["opportunities"] = found["opportunities"][:MAX_OPPORTUNITIES]
    return found


def rank(plan: Plan, ctx: RunContext) -> dict:
    """Rank top-k candidates and draft a decision for each opportunity."""
    found = find_opportunities(plan, ctx)
    policy = plan.policy or DEFAULT_POLICY
    rankings, proposals = {}, {}

    for opp in found["opportunities"]:
        with ctx.trace.step(NAME, f"rank {opp.id}", tool="rank_candidates",
                            opportunity_id=opp.id, k=plan.k) as ev:
            ranked = rank_candidates(opp.id, k=plan.k)
            rankings[opp.id] = ranked
            ev.result_summary = ", ".join(f"{m['student_id']} ({m['score']:.2f})"
                                          for m in ranked) or "no eligible candidates"

        with ctx.trace.step(NAME, f"draft decision {opp.id}", tool="make_decision",
                            opportunity_id=opp.id, policy=policy) as ev:
            record = make_decision(opp.id, policy)
            proposals[opp.id] = record
            ev.result_summary = (f"selected {record.selected or 'nobody'} "
                                 f"({opp.slots} slot(s)), runner-up {record.runner_up}")

    return {"found": found, "policy": policy, "rankings": rankings, "proposals": proposals}


def allocate_all(ctx: RunContext) -> dict:
    """Allocate every opportunity under every policy."""
    allocations = {}
    for policy in POLICIES:
        with ctx.trace.step(NAME, f"allocate ({policy})", tool="allocate", policy=policy) as ev:
            allocations[policy] = allocate(policy)
            filled = sum(len(a["student_ids"]) for a in allocations[policy])
            ev.result_summary = f"{len(allocations[policy])} opportunities, {filled} slots filled"
    return allocations
