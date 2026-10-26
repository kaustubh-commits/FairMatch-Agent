import pytest

from tests.conftest import make_student
from tools.fairness import fairness_metrics, gini


def test_gini_equal_is_zero():
    assert gini([2, 2, 2, 2]) == 0.0


def test_gini_all_zero_is_zero():
    assert gini([0, 0, 0]) == 0.0
    assert gini([]) == 0.0


def test_gini_concentrated_is_high():
    # One person has everything among 4: G = (n-1)/n = 0.75.
    assert gini([0, 0, 0, 10]) == pytest.approx(0.75)


def test_gini_order_does_not_matter():
    assert gini([5, 1, 3]) == gini([1, 3, 5])


def test_fairness_metrics_hand_checked():
    students = [make_student("A", past=2), make_student("B", past=0),
                make_student("C", past=0), make_student("D", past=0)]
    assignments = [
        {"opportunity_id": "O1", "student_ids": ["A"], "scores": {"A": 0.9}},
        {"opportunity_id": "O2", "student_ids": ["B", "C"], "scores": {"B": 0.6, "C": 0.3}},
    ]
    m = fairness_metrics(assignments, students)
    assert m["avg_fit"] == 0.6                      # (0.9 + 0.6 + 0.3) / 3
    assert m["unique_recipients"] == 3
    assert m["repeat_selection_rate"] == round(1 / 3, 3)   # only A had history
    assert m["slots_filled"] == 3
    assert m["gini"] == gini([3, 1, 1, 0])          # past + new per student


def test_fairness_metrics_empty():
    m = fairness_metrics([], [make_student("A")])
    assert m == {"avg_fit": 0.0, "gini": 0.0, "unique_recipients": 0,
                 "repeat_selection_rate": 0.0, "slots_filled": 0}
