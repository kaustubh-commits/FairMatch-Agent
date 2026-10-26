"""
Load real Bangalore internships from the Kaggle Internshala CSV
(dataset `aayush9119/internshala-dataset`), or fall back to a synthetic set.

Steps (PLAN.md 3a):
  1. find the CSV in data/raw/ (any file with "internshala" in its name)
  2. keep only Bangalore / Bengaluru rows (all spellings)
  3. drop duplicates (same title + company)
  4. derive required/preferred skills and a domain from the title
  5. give each listing an area + lat/lon (from the location text if it names
     an area, otherwise a seeded random area)
  6. keep at most MAX_OPPORTUNITIES rows

Everything is seeded, so running it twice gives identical output.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from data_pipeline.constants import (
    BANGALORE_AREAS, CITY_SPELLINGS, DOMAIN_PREFERRED, DOMAIN_PRIORITY,
    KEYWORD_TO_SKILL, SKILL_TO_DOMAIN, SKILLS,
)
from data_pipeline.schemas import Opportunity

MAX_OPPORTUNITIES = 80
SEED = 42

# Different versions of the dataset name their columns differently, so for
# each field we list the column names we are willing to accept.
COLUMN_CANDIDATES = {
    "title": ["internship", "title", "profile", "internship_title", "job_title", "role"],
    "company": ["company", "company_name", "company name"],
    "location": ["location", "locations", "city"],
    "duration": ["duration"],
    "stipend": ["stipend", "salary"],
    "skills": ["skills", "skill", "key_skills"],
}


# ---------------------------------------------------------------- helpers ---

def find_internshala_csv(raw_dir: Path) -> Optional[Path]:
    """Return the first CSV in raw_dir whose name contains 'internshala'."""
    matches = sorted(p for p in Path(raw_dir).glob("*.csv") if "internshala" in p.name.lower())
    return matches[0] if matches else None


def _pick_column(df: pd.DataFrame, field: str) -> Optional[str]:
    """Find which real column in df holds `field` (case-insensitive)."""
    lower = {c.lower().strip(): c for c in df.columns}
    for candidate in COLUMN_CANDIDATES[field]:
        if candidate in lower:
            return lower[candidate]
    return None


def is_bangalore(location: str) -> bool:
    """True if the location text mentions Bangalore in any common spelling."""
    text = str(location).lower()
    return any(spelling in text for spelling in CITY_SPELLINGS)


def derive_skills(title: str, extra_text: str = "") -> tuple[list[str], list[str], str]:
    """
    Turn a job title (plus optional skills text) into
    (required_skills, preferred_skills, domain) using KEYWORD_TO_SKILL.

    Keywords match whole words only, e.g. "ML Intern" -> ML, but
    "JavaScript" does not count as Java.
    """
    text = f"{title} {extra_text}".lower()
    found = set()
    for keyword, skill in KEYWORD_TO_SKILL.items():
        if re.search(r"\b" + re.escape(keyword) + r"\b", text):
            found.add(skill)

    # Keep the order of SKILLS so output is stable; at most 3 required skills.
    required = [s for s in SKILLS if s in found][:3]

    # Domain = the highest-priority domain any required skill points to.
    domains = {SKILL_TO_DOMAIN[s] for s in required}
    domain = next((d for d in DOMAIN_PRIORITY if d in domains), "Other")

    if not required:  # nothing recognised -> a generic, communication-heavy role
        required = ["Communication"]

    preferred = [s for s in DOMAIN_PREFERRED[domain] if s not in required]
    return required, preferred, domain


def parse_duration_weeks(text: str) -> int:
    """'3 Months' -> 12, '6 Weeks' -> 6. Unknown -> 8."""
    text = str(text).lower()
    number = re.search(r"\d+", text)
    if not number:
        return 8
    n = int(number.group())
    return n if "week" in text else n * 4


def parse_stipend(text: str) -> int:
    """'₹ 10,000 /month' -> 10000, '5000-8000 /month' -> 5000, 'Unpaid' -> 0."""
    digits = re.search(r"\d[\d,]*", str(text))
    return int(digits.group().replace(",", "")) if digits else 0


def pick_area(location: str, rng: random.Random) -> str:
    """Use the area named in the location text, else a seeded random area."""
    text = str(location).lower()
    for area in BANGALORE_AREAS:
        if area.lower() in text:
            return area
    return rng.choice(list(BANGALORE_AREAS))


# ------------------------------------------------------------ real data ----

def load_kaggle_opportunities(csv_path: Path, seed: int = SEED,
                              limit: int = MAX_OPPORTUNITIES) -> list[Opportunity]:
    """Read the Internshala CSV and return normalised Bangalore opportunities."""
    df = pd.read_csv(csv_path)
    return opportunities_from_dataframe(df, seed=seed, limit=limit)


def opportunities_from_dataframe(df: pd.DataFrame, seed: int = SEED,
                                 limit: int = MAX_OPPORTUNITIES) -> list[Opportunity]:
    """The real work of the loader, split out so tests can pass a tiny DataFrame."""
    cols = {field: _pick_column(df, field) for field in COLUMN_CANDIDATES}
    if cols["title"] is None or cols["location"] is None:
        raise ValueError(f"CSV needs a title and a location column, found: {list(df.columns)}")

    # 1. Bangalore only.
    df = df[df[cols["location"]].apply(is_bangalore)]

    # 2. Deduplicate on title + company.
    dedupe_on = [c for c in (cols["title"], cols["company"]) if c]
    df = df.drop_duplicates(subset=dedupe_on).head(limit)

    rng = random.Random(seed)
    opportunities = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):

        def get(field: str, default=""):
            """Value of `field` in this row, or default if the column is missing/empty."""
            value = row.get(cols[field]) if cols[field] else None
            return default if value is None or pd.isna(value) else value

        title = str(get("title")).strip()
        skills_text = str(get("skills")) if cols["skills"] else ""
        required, preferred, domain = derive_skills(title, skills_text)
        area = pick_area(get("location"), rng)
        lat, lon = BANGALORE_AREAS[area]

        opportunities.append(Opportunity(
            id=f"O{i:03d}",
            title=title,
            company=str(get("company", "Unknown")).strip() or "Unknown",
            area=area,
            lat=lat,
            lon=lon,
            domain=domain,
            required_skills=required,
            preferred_skills=preferred,
            duration_weeks=parse_duration_weeks(get("duration")),
            stipend=parse_stipend(get("stipend")),
            slots=rng.randint(1, 3),
            source="kaggle",
        ))
    return opportunities


# -------------------------------------------------------- synthetic data ---

# Fictional company names (no real organisations).
SYNTHETIC_COMPANIES = [
    "Nimbus Analytics", "Kaveri Labs", "Banyan Robotics", "Lalbagh Software",
    "Silkboard Systems", "Cubbon AI", "Indigo Cloudworks", "Ragi Health Tech",
    "Namma Fintech", "Garden City Games", "Vidhana Data", "Tulsi Design Studio",
]

# Titles go through derive_skills exactly like real data.
SYNTHETIC_TITLES = [
    "Machine Learning Intern", "Data Science Intern", "Data Analyst Intern",
    "Web Development Intern", "Frontend Developer Intern (React)",
    "Backend Developer Intern (Java)", "Android App Development Intern",
    "Flutter App Developer Intern", "Cloud / DevOps Intern (AWS)",
    "UI/UX Design Intern", "Graphic Design Intern", "NLP Research Intern",
    "Computer Vision Intern", "Python Developer Intern", "Full Stack Developer Intern",
    "Business Analyst Intern", "Content Writing Intern", "Marketing Intern",
    "Operations Intern", "Deep Learning Intern",
]


def synthetic_opportunities(n: int = 60, seed: int = SEED) -> list[Opportunity]:
    """
    A bundled, seeded set of Bangalore opportunities so the app always runs.
    The first 3 are ML internships in Koramangala (the demo's example query).
    """
    rng = random.Random(seed)
    areas = list(BANGALORE_AREAS)
    opportunities = []
    for i in range(1, n + 1):
        if i <= 3:
            title, area = rng.choice(["Machine Learning Intern", "Deep Learning Intern",
                                      "Computer Vision Intern"]), "Koramangala"
        else:
            title, area = rng.choice(SYNTHETIC_TITLES), rng.choice(areas)
        required, preferred, domain = derive_skills(title)
        lat, lon = BANGALORE_AREAS[area]
        opportunities.append(Opportunity(
            id=f"O{i:03d}",
            title=title,
            company=rng.choice(SYNTHETIC_COMPANIES),
            area=area,
            lat=lat,
            lon=lon,
            domain=domain,
            required_skills=required,
            preferred_skills=preferred,
            duration_weeks=rng.choice([4, 8, 12, 24]),
            stipend=rng.choice([0, 5000, 10000, 15000, 20000, 25000]),
            slots=rng.randint(1, 3),
            source="synthetic",
        ))
    return opportunities


def load_opportunities(raw_dir: Path) -> tuple[list[Opportunity], str]:
    """
    Main entry point. Returns (opportunities, source) where source is
    "kaggle" if the CSV was found and usable, else "synthetic".
    """
    csv_path = find_internshala_csv(raw_dir)
    if csv_path is not None:
        opportunities = load_kaggle_opportunities(csv_path)
        if opportunities:
            return opportunities, "kaggle"
        print(f"[load_kaggle] {csv_path.name} has no Bangalore rows; using synthetic data.")
    return synthetic_opportunities(), "synthetic"
