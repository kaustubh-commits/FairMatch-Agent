"""
Student view: which real listings fit ME, and what should I learn next?

  recommend_for_profile  top-k listings a profile is eligible for, best match
                         first, with the per-skill breakdown and the commute
  skills_to_learn        each MISSING skill, ranked by how many extra listings
                         IN THE STUDENT'S INTERESTS it would make them eligible
                         for (at beginner level); listings it opens in other
                         domains are reported separately as "also opens"

Both reuse score_match, so a student sees exactly the score an allocator would.
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.constants import BANGALORE_AREAS, SKILL_TO_DOMAIN, SKILLS
from data_pipeline.schemas import Opportunity, Student
from tools.scoring import DEFAULT_COMMUTE_WEIGHT, MIN_HOURS, score_match
from tools.store import get_student, load_opportunities, load_students

DEFAULT_HOURS = 10      # assumed weekly availability when only skills are given
LEARN_LEVEL = 1         # beginner level already counts towards eligibility
DEFAULT_K = 5


def profile_student(skills: dict[str, int], hours: int = DEFAULT_HOURS,
                    home_area: Optional[str] = None, interests: Optional[list[str]] = None,
                    goal: str = "internship", student_id: str = "YOU") -> Student:
    """A temporary Student built from a form or a chat message (never saved)."""
    levels = {s: max(0, min(3, int(skills.get(s, 0)))) for s in SKILLS}
    lat, lon = BANGALORE_AREAS.get(home_area, (0.0, 0.0))
    return Student(id=student_id, branch="-", year=0,
                   home_area=home_area if home_area in BANGALORE_AREAS else "-", lat=lat, lon=lon,
                   skills=levels, interests=list(interests or []), goal=goal,
                   hours_per_week=hours, past_opportunities=0, source="profile")


def interest_domains(student: Student) -> set[str]:
    """
    The domains a student cares about: their stated interests (+ "Other" if
    their goal is leadership). With no interests stated, use the domains of
    the skills they already have.
    """
    domains = set(student.interests)
    if student.goal == "leadership":
        domains.add("Other")
    if not domains:
        domains = {SKILL_TO_DOMAIN[s] for s, lvl in student.skills.items() if lvl > 0}
    return domains


def _resolve(skills: Optional[dict], student_id: Optional[str], hours: int,
             students: Optional[list[Student]], home_area: Optional[str] = None,
             interests: Optional[list[str]] = None, goal: str = "internship") -> Student:
    """Use a real student's profile if an id is given, else build one."""
    if student_id:
        return get_student(student_id, students if students is not None else load_students())
    return profile_student(skills or {}, hours, home_area, interests, goal)


def _eligible_ids(student: Student, opportunities: list[Opportunity]) -> set[str]:
    return {o.id for o in opportunities if score_match(student, o)["eligible"]}


def recommend_for_profile(skills: Optional[dict[str, int]] = None, student_id: Optional[str] = None,
                          k: int = DEFAULT_K, hours: int = DEFAULT_HOURS,
                          students: Optional[list[Student]] = None,
                          opportunities: Optional[list[Opportunity]] = None,
                          home_area: Optional[str] = None, interests: Optional[list[str]] = None,
                          goal: str = "internship", max_km: Optional[float] = None) -> dict:
    """
    Top-k eligible listings for a profile (skills dict or an existing student id).

    Returns {student_id, student_label, skills, hours, hours_ok, home_area, interests,
    eligible_count, matches}, where each match is score_match(...) plus the listing's
    title/company/area/domain/stipend. The commute counts with the default weight
    when the home area is known. Ties go to the lower opportunity id.
    """
    opportunities = opportunities if opportunities is not None else load_opportunities()
    student = _resolve(skills, student_id, hours, students, home_area, interests, goal)

    matches = []
    for opp in opportunities:
        m = score_match(student, opp, DEFAULT_COMMUTE_WEIGHT, max_km)
        if m["eligible"]:
            m.update(title=opp.title, company=opp.company, area=opp.area, domain=opp.domain,
                     stipend=opp.stipend, slots=opp.slots)
            matches.append(m)
    matches.sort(key=lambda m: (-m["score"], m["opportunity_id"]))
    return {
        "student_id": student.id,
        "student_label": student.label,
        "skills": {s: lvl for s, lvl in student.skills.items() if lvl > 0},
        "hours": student.hours_per_week,
        "hours_ok": student.hours_per_week >= MIN_HOURS,
        "home_area": student.home_area if student.home_area in BANGALORE_AREAS else None,
        "interests": sorted(interest_domains(student)),
        "eligible_count": len(matches),
        "matches": matches[:k],
    }


def skills_to_learn(skills: Optional[dict[str, int]] = None, student_id: Optional[str] = None,
                    hours: int = DEFAULT_HOURS, top: int = 5,
                    students: Optional[list[Student]] = None,
                    opportunities: Optional[list[Opportunity]] = None,
                    interests: Optional[list[str]] = None, goal: str = "internship") -> dict:
    """
    Missing skills ranked by the ADDITIONAL listings each one unlocks.

    For every skill the student doesn't have (level 0) we pretend they learn it
    to LEARN_LEVEL and count listings that switch from ineligible to eligible,
    split into "in their interests" and "elsewhere".

    Returns {"interests": [...], "next": [...], "also_opens": [...]}:
      next        skills that unlock something IN the student's interests, ranked by
                  that count (then by what they open elsewhere, then SKILLS order)
      also_opens  skills that only unlock listings in other domains
    Each item: {skill, unlocks_in_interests, unlocks_elsewhere, examples_in, examples_elsewhere}.
    """
    opportunities = opportunities if opportunities is not None else load_opportunities()
    student = _resolve(skills, student_id, hours, students, None, interests, goal)
    domains = interest_domains(student)
    before = _eligible_ids(student, opportunities)
    by_id = {o.id: o for o in opportunities}

    results = []
    for skill in SKILLS:
        if student.skills.get(skill, 0) > 0:
            continue
        learnt = profile_student({**student.skills, skill: LEARN_LEVEL}, student.hours_per_week)
        gained = sorted(_eligible_ids(learnt, opportunities) - before)
        inside = [o for o in gained if by_id[o].domain in domains]
        outside = [o for o in gained if by_id[o].domain not in domains]
        if gained:
            results.append({
                "skill": skill,
                "unlocks_in_interests": len(inside),
                "unlocks_elsewhere": len(outside),
                "examples_in": list(dict.fromkeys(by_id[o].title for o in inside))[:3],
                "examples_elsewhere": list(dict.fromkeys(by_id[o].title for o in outside))[:3],
            })
    # Stable sorts: equal counts keep the SKILLS order.
    results.sort(key=lambda r: (-r["unlocks_in_interests"], -r["unlocks_elsewhere"]))
    return {
        "interests": sorted(domains),
        "next": [r for r in results if r["unlocks_in_interests"] > 0][:top],
        "also_opens": sorted((r for r in results if r["unlocks_in_interests"] == 0),
                             key=lambda r: -r["unlocks_elsewhere"])[:top],
    }
