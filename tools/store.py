"""
Loads the processed data files once and keeps them in memory.

If data/processed/*.json doesn't exist yet, the pipeline is run first, so
the tools always work out of the box.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

from data_pipeline.build import PROCESSED_DIR, build_all
from data_pipeline.schemas import Opportunity, Student


def _ensure_built() -> None:
    needed = ["students.json", "opportunities.json", "meta.json"]
    if not all((PROCESSED_DIR / name).exists() for name in needed):
        build_all(verbose=True)


def _read(name: str):
    _ensure_built()
    return json.loads((PROCESSED_DIR / name).read_text())


@lru_cache(maxsize=1)
def load_students() -> list[Student]:
    return [Student.from_dict(d) for d in _read("students.json")]


@lru_cache(maxsize=1)
def load_opportunities() -> list[Opportunity]:
    return [Opportunity.from_dict(d) for d in _read("opportunities.json")]


def data_source() -> str:
    """'kaggle' or 'synthetic': where the opportunities came from."""
    return _read("meta.json")["opportunity_source"]


def get_opportunity(opportunity_id: str,
                    opportunities: Optional[list[Opportunity]] = None) -> Opportunity:
    """Look up one opportunity by id (raises KeyError if it doesn't exist)."""
    for opp in opportunities if opportunities is not None else load_opportunities():
        if opp.id == opportunity_id:
            return opp
    raise KeyError(f"Unknown opportunity id: {opportunity_id}")


def get_student(student_id: str, students: Optional[list[Student]] = None) -> Student:
    """Look up one student by id (raises KeyError if it doesn't exist)."""
    for student in students if students is not None else load_students():
        if student.id == student_id:
            return student
    raise KeyError(f"Unknown student id: {student_id}")
