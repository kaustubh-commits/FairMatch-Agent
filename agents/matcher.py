"""
Matcher: runs the scoring / ranking / allocation tools. No LLM.

  rank intent      -> find opportunities, rank candidates, draft a DecisionRecord
  compare intent   -> allocate all opportunities under each policy, and
                      simulate several rounds of each policy
  recommend intent -> best listings for a student + skills to learn next

Each entry point ends with one plain-language headline for the default trace.
"""

from __future__ import annotations

from collections import Counter

from agents.context import RunContext
from agents.planner import Plan
from agents.trace import shorten
from tools.allocation import POLICIES, allocate
from tools.decisions import make_decision
from tools.recommend import recommend_for_profile, skills_to_learn
from tools.scoring import rank_candidates
from tools.search import find_opportunities_near, search_opportunities
from tools.simulation import DEFAULT_ROUNDS, simulate_rounds
from tools.store import load_opportunities

NAME = "Matcher"
MAX_OPPORTUNITIES = 3        # rank at most this many opportunities per request
DEFAULT_POLICY = "skill_first"   # "best candidates" = best match; Equity Analyst checks fairness


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
    wins = Counter()   # selections so far in THIS request, so the per-student cap holds

    for opp in found["opportunities"]:
        with ctx.trace.step(NAME, f"rank {opp.id}", tool="rank_candidates",
                            opportunity_id=opp.id, k=plan.k, policy=policy, max_km=plan.max_km) as ev:
            ranked = rank_candidates(opp.id, k=plan.k, policy=policy, max_km=plan.max_km)
            rankings[opp.id] = ranked
            ev.result_summary = ", ".join(f"{m['student_label']} ({m['score']:.2f})"
                                          for m in ranked) or "no eligible candidates"

        with ctx.trace.step(NAME, f"draft decision {opp.id}", tool="make_decision",
                            opportunity_id=opp.id, policy=policy) as ev:
            record = make_decision(opp.id, policy, prior_wins=dict(wins), max_km=plan.max_km)
            wins.update(record.selected)
            proposals[opp.id] = record
            skipped = record.constraints.get("skipped", {})
            ev.result_summary = (f"selected {', '.join(record.selected) or 'nobody'} "
                                 f"({opp.slots} slot(s)), runner-up {record.runner_up}"
                                 + (f"; skipped {', '.join(skipped)} (per-student cap)" if skipped else ""))

    opps = found["opportunities"]
    if opps:
        eligible = [r.constraints["eligible_candidates"] for r in proposals.values()]
        picked = sum(len(r.selected) for r in proposals.values())
        what = f"{plan.domain} " if plan.domain else ""
        where = f" in {found['area_used']}" if found["area_used"] else ""
        nearest = " (the nearest area with such listings)" if found.get("fell_back") else ""
        counts = ", ".join(str(n) for n in eligible)
        ctx.trace.headline(NAME, f"Ranked the eligible students ({counts}) for {len(opps)} {what}listing"
                                 f"{'' if len(opps) == 1 else 's'}{where}{nearest} and proposed "
                                 f"{picked} of them.")
    else:
        ctx.trace.headline(NAME, shorten(found["note"] or "Found no matching listings."))
    return {"found": found, "policy": policy, "rankings": rankings, "proposals": proposals}


def simulate_all(ctx: RunContext, rounds: int = DEFAULT_ROUNDS, max_km=None) -> dict:
    """Run the multi-round simulation for every policy."""
    results = {}
    for policy in POLICIES:
        with ctx.trace.step(NAME, f"simulate {rounds} rounds ({policy})", tool="simulate_rounds",
                            policy=policy, rounds=rounds, max_km=max_km) as ev:
            results[policy] = simulate_rounds(policy, rounds, max_km=max_km)
            first, last = results[policy][0], results[policy][-1]
            ev.result_summary = (f"Gini {first['gini']:.3f} -> {last['gini']:.3f}; "
                                 f"reach {last['cumulative_reach']} students")
    return results


def allocate_all(ctx: RunContext, max_km=None) -> dict:
    """Allocate every opportunity under every policy."""
    allocations = {}
    for policy in POLICIES:
        with ctx.trace.step(NAME, f"allocate ({policy})", tool="allocate", policy=policy,
                            max_km=max_km) as ev:
            allocations[policy] = allocate(policy, max_km=max_km)
            filled = sum(len(a["student_ids"]) for a in allocations[policy])
            ev.result_summary = f"{len(allocations[policy])} opportunities, {filled} slots filled"
    return allocations


def compare(plan: Plan, ctx: RunContext) -> tuple[dict, dict]:
    """Allocate under every policy, then replay each for several rounds."""
    allocations = allocate_all(ctx, plan.max_km)
    sims = simulate_all(ctx, max_km=plan.max_km)
    limit = f" with a {plan.max_km:g} km commute limit" if plan.max_km else ""
    ctx.trace.headline(NAME, f"Allocated all {len(load_opportunities())} listings under 3 policies{limit} "
                             f"and replayed each for {DEFAULT_ROUNDS} rounds.")
    return allocations, sims


def recommend(plan: Plan, ctx: RunContext) -> dict:
    """Top listings for the student's profile, and which skills would open up the most."""
    who = {"student_id": plan.student_id} if plan.student_id else {"skills": plan.skills}
    with ctx.trace.step(NAME, "match listings to profile", tool="recommend_for_profile",
                        k=plan.k, hours=plan.hours, home_area=plan.area, max_km=plan.max_km, **who) as ev:
        result = recommend_for_profile(skills=plan.skills, student_id=plan.student_id, k=plan.k,
                                       hours=plan.hours, home_area=plan.area, interests=plan.interests,
                                       goal=plan.goal, max_km=plan.max_km)
        ev.result_summary = (f"eligible for {result['eligible_count']} listings; top: "
                             + (", ".join(f"{m['opportunity_id']} ({m['score']:.2f})"
                                          for m in result["matches"]) or "none"))
    with ctx.trace.step(NAME, "find skills to learn", tool="skills_to_learn",
                        interests=plan.interests, **who) as ev:
        learn = skills_to_learn(skills=plan.skills, student_id=plan.student_id, hours=plan.hours,
                                interests=plan.interests, goal=plan.goal)
        ev.result_summary = (", ".join(f"{r['skill']} (+{r['unlocks_in_interests']} in interests)"
                                       for r in learn["next"]) or "nothing new in their interests")

    n = result["eligible_count"]
    text = f"Found {n} listing{'' if n == 1 else 's'} this profile qualifies for"
    if learn["next"]:
        top = learn["next"][0]
        text += (f"; learning {top['skill']} would open {top['unlocks_in_interests']} more in "
                 f"{', '.join(learn['interests'])}")
    ctx.trace.headline(NAME, text + ".")
    return {**result, "learn": learn}
