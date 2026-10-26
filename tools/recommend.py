"""
Student view: which real listings fit ME, and what should I learn next?

  recommend_for_profile  top-k listings a skill profile is eligible for,
                         best fit first, with the usual per-skill breakdown
  skills_to_learn        each MISSING skill, ranked by how many extra listings
                         the student would become eligible for if they learnt
                         it to beginner level (level 1)

Both reuse score_match, so a student sees exactly the score an allocator would.
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.constants import SKILLS
from data_pipeline.schemas import Opportunity, Student
from tools.scoring import MIN_HOURS, score_match
from tools.store import get_student, load_opportunities, load_students

DEFAULT_HOURS = 10      # assumed weekly availability when only skills are given
LEARN_LEVEL = 1         # beginner level already counts towards eligibility
DEFAULT_K = 5


def profile_student(skills: dict[str, int], hours: int = DEFAULT_HOURS,
                    student_id: str = "YOU") -> Student:
    """A temporary Student built from a skill profile (never saved)."""
    levels = {s: max(0, min(3, int(skills.get(s, 0)))) for s in SKILLS}
    return Student(id=student_id, branch="-", year=0, home_area="-", lat=0.0, lon=0.0,
                   skills=levels, interests=[], goal="internship",
                   hours_per_week=hours, past_opportunities=0, source="profile")


def _resolve(skills: Optional[dict], student_id: Optional[str], hours: int,
             students: Optional[list[Student]]) -> Student:
    """Use a real student's profile if an id is given, else build one from skills."""
    if student_id:
        return get_student(student_id, students if students is not None else load_students())
    return profile_student(skills or {}, hours)


def _eligible_ids(student: Student, opportunities: list[Opportunity]) -> set[str]:
    return {o.id for o in opportunities if score_match(student, o)["eligible"]}


def recommend_for_profile(skills: Optional[dict[str, int]] = None, student_id: Optional[str] = None,
                          k: int = DEFAULT_K, hours: int = DEFAULT_HOURS,
                          students: Optional[list[Student]] = None,
                          opportunities: Optional[list[Opportunity]] = None) -> dict:
    """
    Top-k eligible listings for a profile (skills dict or an existing student id).

    Returns {student_id, skills, hours, hours_ok, eligible_count, matches}, where each
    match is score_match(...) plus the listing's title/company/area/domain/stipend.
    Ties go to the lower opportunity id so the result is deterministic.
    """
    opportunities = opportunities if opportunities is not None else load_opportunities()
    student = _resolve(skills, student_id, hours, students)

    matches = []
    for opp in opportunities:
        m = score_match(student, opp)
        if m["eligible"]:
            m.update(title=opp.title, company=opp.company, area=opp.area, domain=opp.domain,
                     stipend=opp.stipend, slots=opp.slots)
            matches.append(m)
    matches.sort(key=lambda m: (-m["score"], m["opportunity_id"]))
    return {
        "student_id": student.id,
        "skills": {s: lvl for s, lvl in student.skills.items() if lvl > 0},
        "hours": student.hours_per_week,
        "hours_ok": student.hours_per_week >= MIN_HOURS,
        "eligible_count": len(matches),
        "matches": matches[:k],
    }


def skills_to_learn(skills: Optional[dict[str, int]] = None, student_id: Optional[str] = None,
                    hours: int = DEFAULT_HOURS, top: int = 5,
                    students: Optional[list[Student]] = None,
                    opportunities: Optional[list[Opportunity]] = None) -> list[dict]:
    """
    Missing skills ranked by how many ADDITIONAL listings each one unlocks.

    For every skill the student doesn't have (level 0) we pretend they learn it
    to LEARN_LEVEL and count listings that switch from ineligible to eligible.
    Skills that unlock nothing are left out. Ties keep the order of SKILLS.
    Returns [{skill, unlocks, examples (up to 3 titles)}].
    """
    opportunities = opportunities if opportunities is not None else load_opportunities()
    student = _resolve(skills, student_id, hours, students)
    before = _eligible_ids(student, opportunities)
    titles = {o.id: o.title for o in opportunities}

    results = []
    for skill in SKILLS:
        if student.skills.get(skill, 0) > 0:
            continue
        learnt = profile_student({**student.skills, skill: LEARN_LEVEL}, student.hours_per_week)
        gained = sorted(_eligible_ids(learnt, opportunities) - before)
        if gained:
            results.append({"skill": skill, "unlocks": len(gained),
                            "examples": [titles[o] for o in gained[:3]]})
    results.sort(key=lambda r: -r["unlocks"])   # stable: equal counts keep SKILLS order
    return results[:top]
