"""
Search tools: filter students and opportunities, and find opportunities
near an area (with a nearest-area fallback so queries never dead-end).
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.constants import BANGALORE_AREAS, distance_km
from data_pipeline.schemas import Opportunity, Student
from tools.store import load_opportunities, load_students


def _same(a: str, b: str) -> bool:
    """Case-insensitive string comparison."""
    return str(a).strip().lower() == str(b).strip().lower()


def search_students(filters: Optional[dict] = None,
                    students: Optional[list[Student]] = None) -> list[Student]:
    """
    Filter students. Supported filters (all optional):
      area        home area, e.g. "Koramangala"
      branch      e.g. "CSE"
      skill       a skill name; combined with min_level (default 1)
      min_level   minimum level for `skill`
      interest    e.g. "ML"
      max_past    at most this many past opportunities
      min_hours   at least this many hours per week
    """
    filters = filters or {}
    result = students if students is not None else load_students()

    if "area" in filters:
        result = [s for s in result if _same(s.home_area, filters["area"])]
    if "branch" in filters:
        result = [s for s in result if _same(s.branch, filters["branch"])]
    if "skill" in filters:
        min_level = filters.get("min_level", 1)
        result = [s for s in result if s.skills.get(filters["skill"], 0) >= min_level]
    if "interest" in filters:
        result = [s for s in result if any(_same(i, filters["interest"]) for i in s.interests)]
    if "max_past" in filters:
        result = [s for s in result if s.past_opportunities <= filters["max_past"]]
    if "min_hours" in filters:
        result = [s for s in result if s.hours_per_week >= filters["min_hours"]]
    return list(result)


def search_opportunities(filters: Optional[dict] = None,
                         opportunities: Optional[list[Opportunity]] = None) -> list[Opportunity]:
    """
    Filter opportunities. Supported filters (all optional):
      area      e.g. "Koramangala"
      domain    e.g. "ML"
      keyword   text that must appear in the title
      skill     a skill that must be in required_skills
    """
    filters = filters or {}
    result = opportunities if opportunities is not None else load_opportunities()

    if "area" in filters:
        result = [o for o in result if _same(o.area, filters["area"])]
    if "domain" in filters:
        result = [o for o in result if _same(o.domain, filters["domain"])]
    if "keyword" in filters:
        result = [o for o in result if filters["keyword"].lower() in o.title.lower()]
    if "skill" in filters:
        result = [o for o in result if any(_same(s, filters["skill"]) for s in o.required_skills)]
    return list(result)


def find_opportunities_near(domain: str, area: str,
                            opportunities: Optional[list[Opportunity]] = None) -> dict:
    """
    Opportunities of `domain` in `area`. If there are none there, fall back to
    the NEAREST area that has some, and say so in `note`.

    Returns {opportunities, requested_area, area_used, fell_back, distance_km, note}.
    `opportunities` is empty only if the domain exists nowhere in the data.
    """
    in_domain = search_opportunities({"domain": domain}, opportunities)
    result = {"opportunities": [], "requested_area": area, "area_used": None,
              "fell_back": False, "distance_km": 0.0, "note": ""}

    if not in_domain:
        result["note"] = f"No {domain} opportunities anywhere in the data."
        return result

    # Exact area first.
    here = [o for o in in_domain if _same(o.area, area)]
    if here:
        result.update(opportunities=here, area_used=here[0].area)
        return result

    # Find the requested area's coordinates (case-insensitive).
    known = {name.lower(): name for name in BANGALORE_AREAS}
    if area.lower() not in known:
        result.update(opportunities=in_domain, fell_back=True,
                      note=f"Unknown area '{area}'; showing all {domain} opportunities.")
        return result
    origin = BANGALORE_AREAS[known[area.lower()]]

    # Nearest area (by distance between area centres) that has this domain.
    areas_with_domain = sorted({o.area for o in in_domain})
    nearest = min(areas_with_domain, key=lambda a: distance_km(origin, BANGALORE_AREAS[a]))
    km = round(distance_km(origin, BANGALORE_AREAS[nearest]), 1)
    result.update(
        opportunities=[o for o in in_domain if o.area == nearest],
        area_used=nearest, fell_back=True, distance_km=km,
        note=f"No {domain} opportunities in {known[area.lower()]}; "
             f"using nearest area {nearest} ({km} km away).",
    )
    return result
