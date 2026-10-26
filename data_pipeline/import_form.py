"""
Import real students from a Google Form CSV export (PLAN.md 3c).

Expected columns (header text may vary slightly, matching is lenient):
  Anonymous ID | Branch | Year | Home area in Bangalore | <one column per skill,
  values Beginner/Intermediate/Advanced/None> | Interests | Goal |
  Hours available per week | Internships/projects received so far | Consent

Rows without consent are dropped. The form collects no names, phones or emails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from data_pipeline.constants import BANGALORE_AREAS, SKILLS
from data_pipeline.schemas import Student

LEVELS = {"none": 0, "beginner": 1, "intermediate": 2, "advanced": 3}
DEFAULT_AREA = "MG Road"  # used when the home area isn't one we know


def _find_column(df: pd.DataFrame, *keywords: str) -> str:
    """
    Return the column whose header contains a keyword (case-insensitive).
    Keywords are tried in order, so put the most specific one first.
    """
    for keyword in keywords:
        for col in df.columns:
            if keyword.lower() in col.lower():
                return col
    raise KeyError(f"No column matching {keywords} in {list(df.columns)}")


def _to_int(value, default: int = 0) -> int:
    """'3+' -> 3, '12 hours' -> 12, '' -> default."""
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else default


def _has_consent(value) -> bool:
    """Google Forms writes the checkbox label when ticked and leaves it empty otherwise."""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text not in ("", "no", "false", "0")


def _match_area(text: str) -> str:
    """Map free text like 'koramangala 5th block' to a known area."""
    lower = str(text).lower()
    for area in BANGALORE_AREAS:
        if area.lower() in lower:
            return area
    return DEFAULT_AREA


def students_from_form(df: pd.DataFrame) -> list[Student]:
    """Convert a form DataFrame into Student records (consenting rows only)."""
    consent_col = _find_column(df, "consent")
    df = df[df[consent_col].apply(_has_consent)]

    cols = {
        "id": _find_column(df, "anonymous id", "id"),
        "branch": _find_column(df, "branch"),
        "year": _find_column(df, "year"),
        "area": _find_column(df, "home area", "area"),
        "interests": _find_column(df, "interest"),
        "goal": _find_column(df, "goal"),
        "hours": _find_column(df, "hours"),
        "past": _find_column(df, "received", "so far"),
    }
    skill_cols = {skill: _find_column(df, skill) for skill in SKILLS}

    students = []
    for _, row in df.iterrows():
        area = _match_area(row[cols["area"]])
        lat, lon = BANGALORE_AREAS[area]
        interests_text = "" if pd.isna(row[cols["interests"]]) else str(row[cols["interests"]])
        students.append(Student(
            id=f"F-{str(row[cols['id']]).strip()}",
            branch=str(row[cols["branch"]]).strip(),
            year=_to_int(row[cols["year"]], default=1),
            home_area=area,
            lat=lat,
            lon=lon,
            skills={s: LEVELS.get(str(row[c]).strip().lower(), 0) for s, c in skill_cols.items()},
            interests=[i.strip() for i in interests_text.split(",") if i.strip()],
            goal=str(row[cols["goal"]]).strip().lower(),
            hours_per_week=_to_int(row[cols["hours"]]),
            past_opportunities=min(3, _to_int(row[cols["past"]])),
            source="form",
        ))
    return students


def import_form(csv_path: Path) -> list[Student]:
    """Read a Google Form CSV export and return Student records."""
    return students_from_form(pd.read_csv(csv_path))
