"""
Seeded generator for ~150 synthetic Bangalore engineering students (PLAN.md 3b).

No real people: colleges are "College A".."College H", ids are S001, S002, ...
Names ("Ananya R.") are generated from a list of common Indian first names
and a last initial, purely so the UI is easier to read. They are synthetic.

The history is deliberately skewed: about 15% of students are
"well-connected" and hold most past opportunities. They also tend to have
slightly stronger-looking profiles, which is exactly why a pure
skill-first policy keeps picking them. That makes the fairness problem
visible in the data.
"""

from __future__ import annotations

import random

from data_pipeline.constants import BANGALORE_AREAS, SKILLS
from data_pipeline.schemas import Student

SEED = 7
N_STUDENTS = 150
CONNECTED_SHARE = 0.15

COLLEGES = [f"College {c}" for c in "ABCDEFGH"]
BRANCHES = ["CSE", "ISE", "ECE", "EEE", "AIML", "Mechanical", "Civil"]
INTEREST_OPTIONS = ["ML", "Data", "Web", "App", "Cloud", "Design", "Other"]

# Common Indian first names from across regions, mixed genders. Combined with a
# random last initial they make readable, obviously synthetic labels.
FIRST_NAMES = [
    "Aarav", "Ananya", "Vihaan", "Diya", "Arjun", "Ishita", "Kabir", "Meera", "Rohan", "Saanvi",
    "Aditya", "Kavya", "Siddharth", "Nisha", "Rahul", "Pooja", "Karthik", "Lakshmi", "Varun", "Sneha",
    "Nikhil", "Priya", "Harsh", "Divya", "Manish", "Anjali", "Suresh", "Deepa", "Vikram", "Shreya",
    "Arnav", "Tanvi", "Pranav", "Gauri", "Imran", "Sana", "Joseph", "Mary", "Gurpreet", "Simran",
    "Tenzin", "Pema", "Abhishek", "Bhavana", "Chetan", "Farah", "Ganesh", "Hema", "Irfan", "Jaya",
    "Kiran", "Lavanya", "Mohan", "Nandini", "Omkar", "Padma", "Ravi", "Swathi", "Tarun", "Uma",
]
LAST_INITIALS = list("ABCDGHIJKMNPRSTVY")
NAME_SEED = 1234   # separate seed: adding names doesn't change any other generated value
GOALS = ["internship", "research", "hackathons", "leadership"]

# Which skills a student who is interested in a domain is likely to have.
INTEREST_SKILLS = {
    "ML": ["ML", "Python", "Data analysis"],
    "Data": ["Data analysis", "Python"],
    "Web": ["Web dev", "Java"],
    "App": ["App dev", "Java"],
    "Cloud": ["Cloud/DevOps", "Python"],
    "Design": ["Design"],
    "Other": ["Communication", "Leadership"],
}


def _make_skills(interests: list[str], connected: bool, rng: random.Random) -> dict[str, int]:
    """Skill levels 0-3: strong in interest areas, weak elsewhere."""
    focus = {skill for i in interests for skill in INTEREST_SKILLS[i]}
    skills = {}
    for skill in SKILLS:
        if skill in focus:
            level = rng.choice([1, 2, 2, 3])
            if connected:  # well-connected students look a bit stronger on paper
                level = min(3, level + 1)
        else:
            level = rng.choice([0, 0, 0, 1])
        skills[skill] = level
    # Everyone has some soft skills.
    skills["Communication"] = max(skills["Communication"], rng.choice([1, 1, 2]))
    return skills


def generate_students(n: int = N_STUDENTS, seed: int = SEED) -> list[Student]:
    """Return n seeded synthetic students with a skewed opportunity history."""
    rng = random.Random(seed)
    n_connected = round(n * CONNECTED_SHARE)
    # Choose which students are "well-connected" (random positions, seeded).
    connected_ids = set(rng.sample(range(n), n_connected))

    areas = list(BANGALORE_AREAS)
    students = []
    for i in range(n):
        connected = i in connected_ids
        area = rng.choice(areas)
        lat, lon = BANGALORE_AREAS[area]
        interests = rng.sample(INTEREST_OPTIONS, rng.randint(1, 3))

        if connected:
            past = rng.randint(3, 6)
        else:
            past = 1 if rng.random() < 0.2 else 0

        students.append(Student(
            id=f"S{i + 1:03d}",
            branch=rng.choice(BRANCHES),
            year=rng.randint(1, 4),
            home_area=area,
            # Small jitter so students in the same area don't stack on the map.
            lat=round(lat + rng.uniform(-0.01, 0.01), 5),
            lon=round(lon + rng.uniform(-0.01, 0.01), 5),
            skills=_make_skills(interests, connected, rng),
            interests=interests,
            goal=rng.choice(GOALS),
            hours_per_week=rng.randint(4, 20),
            past_opportunities=past,
            source="synthetic",
        ))
    # Names use their own random generator, so every other field above stays
    # exactly as it was before names were added.
    name_rng = random.Random(NAME_SEED)
    for student in students:
        student.name = f"{name_rng.choice(FIRST_NAMES)} {name_rng.choice(LAST_INITIALS)}."
    return students
