"""
Fixed reference data shared by the pipeline and the tools.

Keeping these in one place makes the whole system deterministic: the same
keyword always maps to the same skill, the same area to the same coordinates.
"""

import math

# ~15 Bangalore tech areas with approximate centre coordinates (lat, lon).
BANGALORE_AREAS: dict[str, tuple[float, float]] = {
    "Koramangala": (12.9352, 77.6245),
    "HSR Layout": (12.9116, 77.6474),
    "Indiranagar": (12.9719, 77.6412),
    "Whitefield": (12.9698, 77.7500),
    "Electronic City": (12.8452, 77.6602),
    "Marathahalli": (12.9591, 77.6974),
    "Bellandur": (12.9304, 77.6784),
    "BTM Layout": (12.9166, 77.6101),
    "JP Nagar": (12.9063, 77.5857),
    "Jayanagar": (12.9250, 77.5938),
    "MG Road": (12.9756, 77.6066),
    "Malleshwaram": (13.0031, 77.5643),
    "Hebbal": (13.0358, 77.5970),
    "Yelahanka": (13.1007, 77.5963),
    "Sarjapur Road": (12.9100, 77.6870),
}

# The 10 skills from the Google Form (PLAN.md 3c). Students rate each 0-3.
SKILLS: list[str] = [
    "Python", "Java", "ML", "Data analysis", "Web dev",
    "App dev", "Cloud/DevOps", "Design", "Communication", "Leadership",
]

# Opportunity domains (used for map colours and search).
DOMAINS: list[str] = ["ML", "Data", "Web", "App", "Cloud", "Design", "Other"]

# Keyword -> skill. A keyword matches a job title only as a whole word or
# phrase (case-insensitive), so "java" does NOT match "javascript".
KEYWORD_TO_SKILL: dict[str, str] = {
    "machine learning": "ML",
    "deep learning": "ML",
    "artificial intelligence": "ML",
    "ai": "ML",
    "ml": "ML",
    "nlp": "ML",
    "computer vision": "ML",
    "data science": "Data analysis",
    "data scientist": "Data analysis",
    "data analyst": "Data analysis",
    "data analysis": "Data analysis",
    "data analytics": "Data analysis",
    "business analyst": "Data analysis",
    "business analytics": "Data analysis",
    "sql": "Data analysis",
    "excel": "Data analysis",
    "python": "Python",
    "django": "Python",
    "java": "Java",
    "spring": "Java",
    "web developer": "Web dev",
    "web development": "Web dev",
    "front end": "Web dev",
    "frontend": "Web dev",
    "back end": "Web dev",
    "backend": "Web dev",
    "full stack": "Web dev",
    "react": "Web dev",
    "node.js": "Web dev",
    "nodejs": "Web dev",
    "javascript": "Web dev",
    "php": "Web dev",
    "android": "App dev",
    "ios": "App dev",
    "flutter": "App dev",
    "mobile app": "App dev",
    "app developer": "App dev",
    "app development": "App dev",
    "cloud": "Cloud/DevOps",
    "devops": "Cloud/DevOps",
    "aws": "Cloud/DevOps",
    "design": "Design",
    "designer": "Design",
    "ui/ux": "Design",
    "ux": "Design",
    "graphic": "Design",
    "marketing": "Communication",
    "content": "Communication",
    "content writing": "Communication",
    "sales": "Communication",
    "business development": "Communication",
    "human resources": "Leadership",
    "hr": "Leadership",
    "operations": "Leadership",
    "management": "Leadership",
}

# Skills that usually come along with a domain (used as "preferred" skills).
DOMAIN_PREFERRED: dict[str, list[str]] = {
    "ML": ["Python", "Data analysis"],
    "Data": ["Python", "Communication"],
    "Web": ["Design", "Cloud/DevOps"],
    "App": ["Java", "Design"],
    "Cloud": ["Python", "Java"],
    "Design": ["Communication", "Web dev"],
    "Other": ["Communication", "Leadership"],
}

# Which domain a skill points to (first matching skill in a title wins).
SKILL_TO_DOMAIN: dict[str, str] = {
    "ML": "ML",
    "Data analysis": "Data",
    "Web dev": "Web",
    "App dev": "App",
    "Cloud/DevOps": "Cloud",
    "Design": "Design",
    "Python": "Data",
    "Java": "Web",
    "Communication": "Other",
    "Leadership": "Other",
}

# How a domain is detected (priority order: ML beats Data beats Web ...).
DOMAIN_PRIORITY: list[str] = ["ML", "Data", "App", "Web", "Cloud", "Design", "Other"]

# All the ways people spell the city in job listings.
CITY_SPELLINGS: list[str] = ["bangalore", "bengaluru", "banglore", "bengalore", "bangaluru"]


def distance_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle (haversine) distance in km between two (lat, lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(h))
