"""
Data schemas for FairMatch (see PLAN.md, section 3 "Schemas").

We use plain Python dataclasses: they are simple, typed, and easy to turn
into JSON with `asdict`. Every record type has `to_dict` / `from_dict` so the
pipeline can save to and load from `data/processed/*.json`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class Student:
    """One student. Skills are rated 0 (none) to 3 (advanced)."""

    id: str
    branch: str
    year: int
    home_area: str
    lat: float
    lon: float
    skills: dict[str, int]          # e.g. {"Python": 3, "ML": 2}
    interests: list[str]            # e.g. ["ML", "Data"]
    goal: str                       # internship | research | hackathons | leadership
    hours_per_week: int
    past_opportunities: int         # internships/projects received so far
    source: str = "synthetic"       # synthetic | form
    name: str = ""                  # display name, e.g. "Ananya R." (synthetic students only)

    @property
    def label(self) -> str:
        """How people see a student: "Ananya R. (S025)", or just the id if there's no name."""
        return f"{self.name} ({self.id})" if self.name else self.id

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Student":
        return cls(**d)


@dataclass
class Opportunity:
    """One internship / project opening in Bangalore."""

    id: str
    title: str
    company: str
    area: str
    lat: float
    lon: float
    domain: str                     # ML | Data | Web | App | Cloud | Design | Other
    required_skills: list[str]
    preferred_skills: list[str]
    duration_weeks: int
    stipend: int                    # rupees per month (0 = unpaid / unknown)
    slots: int                      # how many students can be selected
    source: str = "synthetic"       # kaggle | synthetic

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Opportunity":
        return cls(**d)


@dataclass
class Appeal:
    """A student's appeal against a decision, and how it was resolved."""

    text: str
    student_id: Optional[str] = None
    outcome: Optional[str] = None       # upheld | revised
    explanation: Optional[str] = None


@dataclass
class DecisionRecord:
    """
    A logged allocation decision. Everything an explanation says must come
    from these fields, so the LLM (Phase 2) never invents a score or reason.
    """

    opportunity_id: str
    selected: list[str]
    runner_up: Optional[str]
    policy: str
    score_breakdown: dict[str, dict]    # student_id -> {skill: points, ...}
    reasons: list[str]
    constraints: dict = field(default_factory=dict)
    status: str = "proposed"            # proposed | approved | appealed | revised
    appeal: Optional[Appeal] = None
    id: Optional[str] = None            # filled in by log_decision
    timestamp: Optional[str] = None     # filled in by log_decision

    def to_dict(self) -> dict:
        return asdict(self)  # asdict also converts the nested Appeal

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionRecord":
        d = dict(d)
        if d.get("appeal") is not None:
            d["appeal"] = Appeal(**d["appeal"])
        return cls(**d)
