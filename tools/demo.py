"""
Phase 1 checkpoint demo:

    python -m tools.demo

1. Finds ML internships in Koramangala (or the nearest area that has some).
2. Prints the top 3 candidates for one of them, with a per-skill breakdown.
3. Allocates every opportunity under the 3 policies and prints fairness metrics.
"""

from tools.allocation import POLICIES, allocate
from tools.fairness import fairness_metrics
from tools.scoring import rank_candidates
from tools.search import find_opportunities_near
from tools.store import data_source, load_opportunities, load_students

DOMAIN, AREA, TOP_K = "ML", "Koramangala", 3


def main() -> None:
    students, opportunities = load_students(), load_opportunities()
    print("=" * 72)
    print("FairMatch Bangalore - Phase 1 demo (deterministic tools, no LLM)")
    print(f"Data: {len(opportunities)} opportunities (source: {data_source()}), "
          f"{len(students)} students")
    print("=" * 72)

    # --- 1. Find the opportunity ---------------------------------------------
    found = find_opportunities_near(DOMAIN, AREA)
    if found["note"]:
        print(f"Note: {found['note']}")
    if not found["opportunities"]:
        print("Nothing to rank.")
        return
    opp = found["opportunities"][0]
    print(f"\n{len(found['opportunities'])} {DOMAIN} opportunities in {found['area_used']}. "
          f"Ranking the first one:")
    print(f"  {opp.id}: {opp.title} @ {opp.company} ({opp.area}), {opp.slots} slot(s)")
    print(f"  required: {', '.join(opp.required_skills)} | "
          f"preferred: {', '.join(opp.preferred_skills) or '-'}")

    # --- 2. Top candidates ----------------------------------------------------
    print(f"\nTop {TOP_K} candidates (pure skill fit):")
    for rank, match in enumerate(rank_candidates(opp.id, k=TOP_K), start=1):
        parts = ", ".join(f"{skill} {b['level']}/3 ({b['kind'][:3]})"
                          for skill, b in match["breakdown"].items())
        print(f"  {rank}. {match['student_id']}  fit {match['score']:.2f}  "
              f"past opps {match['past_opportunities']}  | {parts}")

    # --- 3. Policy comparison ---------------------------------------------------
    print("\nFairness metrics, all opportunities allocated under each policy:")
    header = f"  {'policy':<14}{'avg fit':>9}{'gini':>8}{'unique':>9}{'repeat %':>10}{'slots':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for policy in POLICIES:
        m = fairness_metrics(allocate(policy))
        print(f"  {policy:<14}{m['avg_fit']:>9.3f}{m['gini']:>8.3f}{m['unique_recipients']:>9}"
              f"{m['repeat_selection_rate'] * 100:>9.1f}%{m['slots_filled']:>7}")
    print("\n  gini = inequality of total opportunities per student (lower = fairer)")
    print("  repeat % = share of slots going to students who already had one")


if __name__ == "__main__":
    main()
