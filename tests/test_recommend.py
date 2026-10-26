from data_pipeline.constants import SKILLS
from tests.conftest import make_opportunity, make_student
from tools.recommend import profile_student, recommend_for_profile, skills_to_learn


def opps():
    return [
        make_opportunity("O1", ["ML"], ["Python"]),
        make_opportunity("O2", ["Python"], ["Data analysis"], domain="Data"),
        make_opportunity("O3", ["Web dev"], [], domain="Web"),
        make_opportunity("O4", ["Web dev", "Design"], [], domain="Web"),
        make_opportunity("O5", ["Design"], [], domain="Design"),
    ]


def test_profile_student_fills_and_clamps_levels():
    s = profile_student({"Python": 5, "ML": -1})
    assert s.skills["Python"] == 3 and s.skills["ML"] == 0 and set(s.skills) == set(SKILLS)


def test_recommend_orders_by_fit_and_only_eligible():
    result = recommend_for_profile({"Python": 3, "ML": 1}, opportunities=opps())
    ids = [m["opportunity_id"] for m in result["matches"]]
    # O2 (Python required, 3/3) fits best; O1 needs ML (1/3) + Python preferred; web/design ineligible.
    assert ids == ["O2", "O1"]
    assert result["eligible_count"] == 2 and result["hours_ok"]
    assert result["matches"][0]["title"] == "Data Intern"


def test_recommend_respects_k():
    assert len(recommend_for_profile({"Web dev": 2, "Design": 2, "Python": 2}, k=2,
                                     opportunities=opps())["matches"]) == 2


def test_recommend_too_few_hours_means_no_matches():
    result = recommend_for_profile({"Python": 3}, hours=4, opportunities=opps())
    assert result["matches"] == [] and not result["hours_ok"]


def test_recommend_by_student_id():
    students = [make_student("S1", {"Python": 3})]
    result = recommend_for_profile(student_id="S1", students=students, opportunities=opps())
    assert result["student_id"] == "S1" and result["matches"][0]["opportunity_id"] == "O2"


def test_skills_to_learn_ranks_by_unlocked_listings():
    # With only Python: Web dev unlocks O3 and O4 (half of O4's required skills),
    # Design unlocks O4 and O5, ML unlocks O1. Ties keep SKILLS order (Web dev before Design).
    learn = skills_to_learn({"Python": 2}, opportunities=opps())
    assert [(r["skill"], r["unlocks"]) for r in learn] == [("Web dev", 2), ("Design", 2), ("ML", 1)]
    assert learn[0]["examples"] == ["Web Intern", "Web Intern"]


def test_skills_to_learn_skips_known_and_useless_skills():
    learn = skills_to_learn({"Python": 2, "Web dev": 1}, opportunities=opps())
    skills = [r["skill"] for r in learn]
    assert "Python" not in skills and "Web dev" not in skills and "Java" not in skills
