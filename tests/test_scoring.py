from tests.conftest import make_opportunity, make_student
from tools.recommend import profile_student
from tools.scoring import commute_km, rank_candidates, score_match


def test_perfect_match_scores_one():
    student = make_student("A", {"ML": 3, "Python": 3})
    opp = make_opportunity("O", ["ML"], ["Python"])
    result = score_match(student, opp)
    assert result["score"] == 1.0
    assert result["eligible"]


def test_partial_match_uses_weights():
    # ML required (weight 2) at level 3 -> 2 pts; Python preferred (weight 1) at 0 -> 0 pts.
    # Same area -> commute 0 km -> closeness 1. Score = 0.8 * 0.667 + 0.2 * 1 = 0.734.
    student = make_student("A", {"ML": 3})
    opp = make_opportunity("O", ["ML"], ["Python"])
    result = score_match(student, opp)
    assert result["skill_fit"] == round(2 / 3, 3)
    assert result["commute_km"] == 0.0 and result["closeness"] == 1.0
    assert result["score"] == 0.734
    assert result["breakdown"]["ML"] == {"level": 3, "kind": "required", "weight": 2,
                                         "points": 2.0, "max_points": 2}
    assert result["breakdown"]["Python"]["points"] == 0.0


def test_zero_match_is_ineligible():
    result = score_match(make_student("A"), make_opportunity("O", ["ML"], ["Python"]))
    assert result["skill_fit"] == 0.0 and result["score"] == 0.2      # only closeness counts
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


# ---- commute ------------------------------------------------------------------

def test_commute_breaks_ties_between_equal_skills():
    near = make_student("S2", {"ML": 3}, area="Koramangala")
    far = make_student("S1", {"ML": 3}, area="Yelahanka")
    opp = make_opportunity("O", ["ML"], area="Koramangala")
    ranked = rank_candidates("O", k=2, students=[far, near], opportunities=[opp])
    assert [m["student_id"] for m in ranked] == ["S2", "S1"]
    assert ranked[1]["commute_km"] > 15 and ranked[0]["skill_fit"] == ranked[1]["skill_fit"]


def test_max_km_makes_far_students_ineligible_with_reason():
    far = make_student("S1", {"ML": 3}, area="Yelahanka")
    result = score_match(far, make_opportunity("O", ["ML"], area="Koramangala"), max_km=10)
    assert not result["eligible"]
    assert any("exceeds the 10 km limit" in r for r in result["reasons"])


def test_unknown_home_means_score_is_skill_fit():
    student = profile_student({"ML": 2})               # no home area
    result = score_match(student, make_opportunity("O", ["ML"]))
    assert result["commute_km"] is None and result["score"] == result["skill_fit"]


def test_commute_weight_zero_ignores_distance():
    far = make_student("S1", {"ML": 3}, area="Yelahanka")
    opp = make_opportunity("O", ["ML"], area="Koramangala")
    assert score_match(far, opp, commute_weight=0.0)["score"] == 1.0
    assert commute_km(far, opp) > 15
