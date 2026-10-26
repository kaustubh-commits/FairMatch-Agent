import pytest

from tests.conftest import make_opportunity, make_student
from tools.decisions import (appeal_decision, approve_decision, get_decision,
                             list_decisions, log_decision, make_decision)


@pytest.fixture
def path(tmp_path):
    """Each test gets its own empty decisions file."""
    return tmp_path / "decisions.json"


def test_make_decision_fields(students, opportunities):
    record = make_decision("O1", "skill_first", students, opportunities)
    assert record.selected == ["S1"]
    assert record.runner_up == "S2"
    assert record.policy == "skill_first"
    assert record.score_breakdown["S1"]["score"] == 1.0
    assert record.constraints["slots"] == 1
    assert any(r.startswith("S1 selected") for r in record.reasons)


def test_log_and_get_round_trip(students, opportunities, path):
    saved = log_decision(make_decision("O1", "hybrid", students, opportunities), path)
    assert saved.id == "D0001" and saved.timestamp and saved.status == "proposed"
    loaded = get_decision("D0001", path)
    assert loaded == saved
    assert log_decision(make_decision("O2", "hybrid", students, opportunities), path).id == "D0002"


def test_list_decisions(students, opportunities, path):
    assert list_decisions(path) == []
    log_decision(make_decision("O1", "hybrid", students, opportunities), path)
    log_decision(make_decision("O2", "hybrid", students, opportunities), path)
    assert [r.id for r in list_decisions(path)] == ["D0001", "D0002"]


def test_get_missing_raises(path):
    with pytest.raises(KeyError):
        get_decision("D9999", path)


def test_approve(students, opportunities, path):
    log_decision(make_decision("O1", "hybrid", students, opportunities), path)
    assert approve_decision("D0001", path).status == "approved"
    assert get_decision("D0001", path).status == "approved"


def test_appeal_upheld_when_gap_is_large(students, opportunities, path):
    # skill_first picks S1 (fit 1.0). S3 (fit 0.33) appeals: gap too large.
    log_decision(make_decision("O1", "skill_first", students, opportunities), path)
    record = appeal_decision("D0001", "I am keen to learn", student_id="S3",
                             path=path, students=students, opportunities=opportunities)
    assert record.status == "appealed"
    assert record.appeal.outcome == "upheld"
    assert record.selected == ["S1"]
    assert "Decision upheld" in record.appeal.explanation


def test_appeal_revised_when_close_and_fewer_past(path):
    # Points possible: 2 + 2 (required) + 1 + 1 + 1 (preferred) = 7.
    # S1 has everything at 3 -> fit 1.0, 4 past.
    # S2 has Leadership 2 -> loses 1/3 pt -> fit 6.667/7 = 0.952, 0 past.
    # Gap 0.048 <= 0.05 and S2 has fewer past opportunities -> revised.
    full = {"ML": 3, "Python": 3, "Data analysis": 3, "Communication": 3, "Leadership": 3}
    students = [make_student("S1", full, past=4),
                make_student("S2", {**full, "Leadership": 2}, past=0)]
    opps = [make_opportunity("O1", ["ML", "Python"], ["Data analysis", "Communication", "Leadership"])]
    log_decision(make_decision("O1", "skill_first", students, opps), path)

    record = appeal_decision("D0001", "I have had no opportunities yet",
                             path=path, students=students, opportunities=opps)  # runner-up appeals
    assert record.appeal.student_id == "S2"
    assert record.appeal.outcome == "revised"
    assert record.status == "revised"
    assert record.selected == ["S2"]
    assert get_decision("D0001", path).selected == ["S2"]   # persisted


def test_appeal_by_ineligible_student_is_upheld(students, opportunities, path):
    log_decision(make_decision("O1", "skill_first", students, opportunities), path)
    record = appeal_decision("D0001", "please", student_id="S4", path=path,
                             students=students, opportunities=opportunities)
    assert record.appeal.outcome == "upheld"
    assert "not eligible" in record.appeal.explanation
