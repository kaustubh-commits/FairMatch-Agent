"""
Small hand-made data shared by all tests (pytest loads this file automatically).

Using tiny fixtures instead of the real data files means every expected
number in the tests can be checked by hand.
"""

import pytest

from data_pipeline.constants import BANGALORE_AREAS, SKILLS
from data_pipeline.schemas import Opportunity, Student


def make_student(sid, skills=None, past=0, hours=10, area="Koramangala", interests=None):
    """A student with every skill at 0 unless given in `skills`."""
    lat, lon = BANGALORE_AREAS[area]
    levels = {s: 0 for s in SKILLS}
    levels.update(skills or {})
    return Student(id=sid, branch="CSE", year=3, home_area=area, lat=lat, lon=lon,
                   skills=levels, interests=interests or ["ML"], goal="internship",
                   hours_per_week=hours, past_opportunities=past)


def make_opportunity(oid, required, preferred=(), area="Koramangala", domain="ML", slots=1):
    lat, lon = BANGALORE_AREAS[area]
    return Opportunity(id=oid, title=f"{domain} Intern", company="TestCo", area=area,
                       lat=lat, lon=lon, domain=domain, required_skills=list(required),
                       preferred_skills=list(preferred), duration_weeks=8, stipend=10000,
                       slots=slots)


@pytest.fixture
def students():
    return [
        make_student("S1", {"ML": 3, "Python": 3}, past=4),        # strong, well-connected
        make_student("S2", {"ML": 3, "Python": 2}, past=0),        # strong, newcomer
        make_student("S3", {"ML": 1, "Python": 1}, past=0),        # weaker newcomer
        make_student("S4", {"ML": 3, "Python": 3}, past=0, hours=4),  # strong but no time
        make_student("S5", {"Web dev": 3}, past=1, area="Whitefield", interests=["Web"]),
    ]


@pytest.fixture
def opportunities():
    return [
        make_opportunity("O1", ["ML"], ["Python"], slots=1),
        make_opportunity("O2", ["ML"], ["Python"], slots=1, area="HSR Layout"),
        make_opportunity("O3", ["Web dev"], [], area="Whitefield", domain="Web", slots=1),
    ]
