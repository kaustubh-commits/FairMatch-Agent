"""
Grounding check: is an LLM-written sentence backed by the facts we gave it?

The LLM may only REPHRASE facts that came from the tools. Before showing
any LLM text we check that every number and every id (S025, D0001, O049)
in it also appears in the facts. If anything is new, the text is rejected
and the agent uses its template instead.

This is a simple, strict filter: it can reject a harmless sentence, but it
will not let an invented score or student through.
"""

from __future__ import annotations

import re

NUMBER = re.compile(r"\d+(?:\.\d+)?")
ID = re.compile(r"\b(?:[SDO]\d{3,4}|F-[\w-]+)\b")


def _collect(value, numbers: set, texts: list) -> None:
    """Walk nested dicts/lists and gather every number and string."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        numbers.add(float(value))
    elif isinstance(value, str):
        texts.append(value)
        numbers.update(float(n) for n in NUMBER.findall(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            _collect(k, numbers, texts)
            _collect(v, numbers, texts)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _collect(v, numbers, texts)


def _matches(n: float, allowed: set[float]) -> bool:
    """A number is grounded if it equals a fact, possibly rounded or as a percentage."""
    for a in allowed:
        for candidate in (a, a * 100):           # 0.432 may be written as 43.2%
            for digits in range(5):           # exact (tools round to <= 4 dp) or rounded
                if round(candidate, digits) == n:
                    return True
    return False


def ungrounded(text: str, facts) -> list[str]:
    """Return the numbers/ids in `text` that are NOT in `facts` (empty list = grounded)."""
    numbers, texts = set(), []
    _collect(facts, numbers, texts)
    fact_text = " ".join(texts)

    problems = [i for i in ID.findall(text) if i not in fact_text]
    text_without_ids = ID.sub(" ", text)
    problems += [n for n in NUMBER.findall(text_without_ids) if not _matches(float(n), numbers)]
    return problems
