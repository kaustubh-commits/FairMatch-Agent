import pytest

from tests.conftest import make_opportunity, make_student
from tools.allocation import MAX_PER_STUDENT, allocate, priority


def test_slots_respected(students, opportunities):
    for policy in ("skill_first", "equal_access", "hybrid"):
        for a, opp in zip(allocate(policy, students, opportunities), opportunities):
            assert len(a["student_ids"]) <= opp.slots


def test_max_per_student_respected():
    # One strong student, many one-slot opportunities: they can win only MAX_PER_STUDENT.
    students = [make_student("S1", {"ML": 3}), make_student("S2", {"ML": 1})]
    opps = [make_opportunity(f"O{i}", ["ML"]) for i in range(5)]
    assignments = allocate("skill_first", students, opps)
    wins = sum(a["student_ids"].count("S1") for a in assignments)
    assert wins == MAX_PER_STUDENT


def test_skill_first_picks_best_fit(students, opportunities):
    first = allocate("skill_first", students, opportunities)[0]
    assert first["student_ids"] == ["S1"]      # highest fit, despite 4 past opportunities
    assert first["runner_up"] == "S2"


def test_equal_access_favours_low_history(students, opportunities):
    first = allocate("equal_access", students, opportunities)[0]
    assert first["student_ids"] == ["S2"]      # best fit among 0-past students


def test_hybrid_balances_fit_and_access():
    # S1: fit 1.0, past 4 -> 1.0 / 3 = 0.33.  S2: fit 0.78, past 0 -> 0.78. S2 wins.
    students = [make_student("S1", {"ML": 3, "Python": 3}, past=4),
                make_student("S2", {"ML": 3, "Python": 1}, past=0)]
    opps = [make_opportunity("O1", ["ML"], ["Python"])]
    assert allocate("hybrid", students, opps)[0]["student_ids"] == ["S2"]


def test_unknown_policy_raises(students, opportunities):
    with pytest.raises(ValueError):
        allocate("lottery", students, opportunities)
    with pytest.raises(ValueError):
        priority("lottery", 0.5, 0)


def test_allocation_is_deterministic(students, opportunities):
    assert allocate("hybrid", students, opportunities) == allocate("hybrid", students, opportunities)
