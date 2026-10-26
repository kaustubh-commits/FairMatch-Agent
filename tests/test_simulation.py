import pytest

from tests.conftest import make_opportunity, make_student
from tools.simulation import simulate_rounds


def setup():
    students = [make_student("S1", {"ML": 3}, past=3), make_student("S2", {"ML": 2}, past=0),
                make_student("S3", {"ML": 2}, past=0)]
    return students, [make_opportunity("O1", ["ML"], slots=1)]


def test_rounds_and_original_untouched():
    students, opps = setup()
    history = simulate_rounds("skill_first", rounds=4, students=students, opportunities=opps)
    assert [h["round"] for h in history] == [1, 2, 3, 4]
    assert students[0].past_opportunities == 3          # copies were used


def test_skill_first_keeps_picking_the_same_student():
    students, opps = setup()
    history = simulate_rounds("skill_first", rounds=3, students=students, opportunities=opps)
    assert [h["repeat_selection_rate"] for h in history] == [1.0, 1.0, 1.0]   # S1 every round
    assert history[-1]["cumulative_reach"] == 1


def test_equal_access_rotates_because_winners_gain_history():
    students, opps = setup()
    history = simulate_rounds("equal_access", rounds=3, students=students, opportunities=opps)
    # Round 1: S2 (0 past). Round 2: S3 (S2 now has 1). Round 3: S2 again (S2 & S3 tie at 1, S2 wins on fit tie -> id).
    assert [h["repeat_selection_rate"] for h in history] == [0.0, 0.0, 0.0]
    assert history[-1]["cumulative_reach"] == 2


def test_gini_reflects_accumulated_history():
    students, opps = setup()
    sf = simulate_rounds("skill_first", rounds=3, students=students, opportunities=opps)
    ea = simulate_rounds("equal_access", rounds=3, students=students, opportunities=opps)
    assert sf[-1]["gini"] > ea[-1]["gini"]


def test_unknown_policy_raises():
    students, opps = setup()
    with pytest.raises(ValueError):
        simulate_rounds("lottery", students=students, opportunities=opps)
