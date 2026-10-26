from tests.conftest import make_opportunity
from tools.search import find_opportunities_near, search_opportunities, search_students


def test_search_students_filters(students):
    assert [s.id for s in search_students({"skill": "ML", "min_level": 3}, students)] == ["S1", "S2", "S4"]
    assert [s.id for s in search_students({"area": "whitefield"}, students)] == ["S5"]
    assert [s.id for s in search_students({"max_past": 0, "min_hours": 8}, students)] == ["S2", "S3"]
    assert [s.id for s in search_students({"interest": "Web"}, students)] == ["S5"]


def test_search_students_no_filters_returns_all(students):
    assert len(search_students(None, students)) == len(students)


def test_search_opportunities_filters(opportunities):
    assert [o.id for o in search_opportunities({"domain": "ml"}, opportunities)] == ["O1", "O2"]
    assert [o.id for o in search_opportunities({"area": "Koramangala"}, opportunities)] == ["O1"]
    assert [o.id for o in search_opportunities({"skill": "Web dev"}, opportunities)] == ["O3"]
    assert [o.id for o in search_opportunities({"keyword": "web"}, opportunities)] == ["O3"]


def test_find_near_exact_area(opportunities):
    found = find_opportunities_near("ML", "Koramangala", opportunities)
    assert [o.id for o in found["opportunities"]] == ["O1"]
    assert not found["fell_back"] and found["note"] == ""


def test_find_near_falls_back_to_nearest_area():
    # No ML in Koramangala; HSR Layout (~4 km) is nearer than Hebbal (~11 km).
    opps = [make_opportunity("O1", ["ML"], area="Hebbal"),
            make_opportunity("O2", ["ML"], area="HSR Layout")]
    found = find_opportunities_near("ML", "Koramangala", opps)
    assert found["fell_back"]
    assert found["area_used"] == "HSR Layout"
    assert [o.id for o in found["opportunities"]] == ["O2"]
    assert "nearest area HSR Layout" in found["note"]
    assert 0 < found["distance_km"] < 10


def test_find_near_unknown_domain_is_empty(opportunities):
    found = find_opportunities_near("Biotech", "Koramangala", opportunities)
    assert found["opportunities"] == []
    assert "No Biotech" in found["note"]


def test_find_near_unknown_area_returns_all_in_domain(opportunities):
    found = find_opportunities_near("ML", "Atlantis", opportunities)
    assert found["fell_back"]
    assert [o.id for o in found["opportunities"]] == ["O1", "O2"]
