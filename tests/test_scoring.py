from tests.conftest import make_opportunity, make_student
from tools.scoring import rank_candidates, score_match


def test_perfect_match_scores_one():
    student = make_student("A", {"ML": 3, "Python": 3})
    opp = make_opportunity("O", ["ML"], ["Python"])
    result = score_match(student, opp)
    assert result["score"] == 1.0
    assert result["eligible"]


def test_partial_match_uses_weights():
    # ML required (weight 2) at level 3 -> 2 pts; Python preferred (weight 1) at 0 -> 0 pts.
    student = make_student("A", {"ML": 3})
    opp = make_opportunity("O", ["ML"], ["Python"])
    result = score_match(student, opp)
    assert result["score"] == round(2 / 3, 3)
    assert result["breakdown"]["ML"] == {"level": 3, "kind": "required", "weight": 2,
                                         "points": 2.0, "max_points": 2}
    assert result["breakdown"]["Python"]["points"] == 0.0


def test_zero_match_is_ineligible():
    result = score_match(make_student("A"), make_opportunity("O", ["ML"], ["Python"]))
    assert result["score"] == 0.0
    assert not result["eligible"]
    assert any("required skills" in r for r in result["reasons"])


def test_too_few_hours_is_ineligible():
    student = make_student("A", {"ML": 3}, hours=4)
    result = score_match(student, make_opportunity("O", ["ML"]))
    assert not result["eligible"]
    assert any("hours/week" in r for r in result["reasons"])


def test_rank_candidates_order_and_eligibility(students, opportunities):
    ranked = rank_candidates("O1", k=10, students=students, opportunities=opportunities)
    ids = [m["student_id"] for m in ranked]
    # S4 (no time) and S5 (no ML) are ineligible; S1 beats S2 on Python; S3 last.
    assert ids == ["S1", "S2", "S3"]


def test_rank_candidates_ties_broken_by_id():
    students = [make_student("S9", {"ML": 2}), make_student("S1", {"ML": 2})]
    opps = [make_opportunity("O", ["ML"])]
    ranked = rank_candidates("O", k=2, students=students, opportunities=opps)
    assert [m["student_id"] for m in ranked] == ["S1", "S9"]


def test_rank_candidates_ties_prefer_fewer_past_opportunities():
    students = [make_student("S010", {"ML": 2}, past=5), make_student("S140", {"ML": 2}, past=0)]
    ranked = rank_candidates("O", k=2, students=students, opportunities=[make_opportunity("O", ["ML"])])
    assert [m["student_id"] for m in ranked] == ["S140", "S010"]


def test_rank_candidates_respects_k(students, opportunities):
    assert len(rank_candidates("O1", k=1, students=students, opportunities=opportunities)) == 1
