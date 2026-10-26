"""
Planner: turns the user's request into a Plan (intent + parameters + steps).

Two ways to understand the request:
  - LLM mode: the LLM fills in a small JSON form. Every field is checked
    against the allowed values; anything invalid is replaced by the rule-based
    reading, so the LLM can never send the agents somewhere unexpected.
  - demo-safe mode: plain keyword/regex rules (`parse_rules`).

The STEPS for each intent are fixed below. The LLM only chooses which
intent and which parameters, never which tools exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from agents.context import RunContext
from data_pipeline.constants import BANGALORE_AREAS, DOMAINS
from tools.allocation import POLICIES

NAME = "Planner"
INTENTS = ("rank", "compare", "explain", "appeal")
DEFAULT_K = 3
MAX_K = 10

# Fixed tool-call recipes per intent: (agent, tool).
STEPS = {
    "rank": [("Matcher", "find_opportunities_near"), ("Matcher", "rank_candidates"),
             ("Matcher", "make_decision"), ("Equity Analyst", "fairness_metrics"),
             ("Auditor", "score_match"), ("Auditor", "log_decision"),
             ("Explainer", "explain record")],
    "compare": [("Matcher", "allocate"), ("Equity Analyst", "fairness_metrics"),
                ("Auditor", "check allocation rules"), ("Explainer", "metrics table")],
    "explain": [("Auditor", "get_decision"), ("Explainer", "explain record")],
    "appeal": [("Auditor", "appeal_decision"), ("Explainer", "explain record")],
}

# Words that point to a domain (checked in order, whole words).
DOMAIN_WORDS = [
    ("ML", ["ml", "machine learning", "ai", "artificial intelligence", "deep learning",
            "computer vision", "nlp"]),
    ("Data", ["data", "analytics", "analyst", "data science"]),
    ("App", ["app", "android", "ios", "mobile", "flutter"]),
    ("Web", ["web", "frontend", "backend", "full stack", "react"]),
    ("Cloud", ["cloud", "devops", "aws"]),
    ("Design", ["design", "ui/ux", "ux", "graphic"]),
]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


@dataclass
class Plan:
    intent: str
    domain: Optional[str] = None
    area: Optional[str] = None
    k: int = DEFAULT_K
    policy: Optional[str] = None
    decision_id: Optional[str] = None
    student_id: Optional[str] = None
    appeal_reason: Optional[str] = None
    steps: list = field(default_factory=list)
    source: str = "rules"            # "rules" or "llm"

    def params(self) -> dict:
        """The non-empty parameters, for the trace."""
        keys = ("domain", "area", "k", "policy", "decision_id", "student_id", "appeal_reason")
        return {k: getattr(self, k) for k in keys if getattr(self, k) not in (None, "")}


# ------------------------------------------------------------- rules -------

def _has_word(text: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


def parse_rules(request: str) -> Plan:
    """Rule-based reading of the request (used in demo-safe mode and to check the LLM)."""
    text = request.lower()

    decision = re.search(r"\bd\d{4}\b", text)
    student = re.search(r"\b(s\d{3}|f-[\w-]+)\b", text)

    if _has_word(text, "appeal") or _has_word(text, "contest"):
        intent = "appeal"
    elif (decision or _has_word(text, "why was")
          or (_has_word(text, "explain") and _has_word(text, "decision"))):
        intent = "explain"
    elif (any(_has_word(text, w) for w in ("compare", "policy", "policies", "policy lab"))
          and not any(_has_word(text, w) for w in ("candidate", "candidates"))):
        intent = "compare"
    else:
        intent = "rank"

    domain = next((d for d, words in DOMAIN_WORDS if any(_has_word(text, w) for w in words)), None)
    area = next((a for a in BANGALORE_AREAS if a.lower() in text), None)
    policy = next((p for p in POLICIES if p in text or p.replace("_", " ") in text), None)

    k = DEFAULT_K
    count = re.search(r"\b(?:top|best|first)\s+(\d+|" + "|".join(NUMBER_WORDS) + r")\b", text)
    if count:
        word = count.group(1)
        k = int(word) if word.isdigit() else NUMBER_WORDS[word]

    # Appeal reason: whatever follows "because" or the first ":".
    reason = None
    if intent == "appeal":
        match = re.search(r"(?:because|:)\s*(.+)", request, flags=re.IGNORECASE)
        reason = match.group(1).strip() if match else request.strip()

    return Plan(intent=intent, domain=domain, area=area, k=max(1, min(k, MAX_K)),
                policy=policy, decision_id=decision.group().upper() if decision else None,
                student_id=_format_student(student.group()) if student else None,
                appeal_reason=reason)


def _format_student(raw: str) -> str:
    return raw.upper() if raw.lower().startswith("s") else "F-" + raw[2:]


# --------------------------------------------------------------- LLM -------

PROMPT = """You convert a request to an internship-matching system into JSON.
Do NOT answer the request, do NOT invent scores or reasons. Only fill this form.

Return exactly one JSON object with these keys:
  "intent":        one of {intents}
                   rank = find/rank candidates for opportunities;
                   compare = compare allocation policies / fairness;
                   explain = explain an existing decision (needs a decision id like D0001);
                   appeal = a student appeals a decision
  "domain":        one of {domains} or null
  "area":          one of {areas} or null
  "k":             integer 1-{max_k} (how many candidates), default {k}
  "policy":        one of {policies} or null
  "decision_id":   like "D0001" or null
  "student_id":    like "S001" or null
  "appeal_reason": the student's reason in their own words, or null

Request: {request}"""


def _validate_llm(raw: dict, fallback: Plan, request: str = "") -> Plan:
    """
    Keep each LLM field only if it is valid; otherwise use the rule-based value.
    Ids (D0001, S011) must also appear word for word in the request: an LLM once
    "filled in" D0001 for "explain the most recent decision", which is a made-up id.
    """
    def in_request(value) -> bool:
        return str(value).lower() in request.lower()

    def pick(key, ok):
        value = raw.get(key)
        return value if value is not None and ok(value) else getattr(fallback, key)

    areas_lower = {a.lower(): a for a in BANGALORE_AREAS}
    area = raw.get("area")
    area = areas_lower.get(str(area).lower()) if area else None

    plan = Plan(
        intent=pick("intent", lambda v: v in INTENTS),
        domain=pick("domain", lambda v: v in DOMAINS),
        area=area or fallback.area,
        k=pick("k", lambda v: isinstance(v, int) and 1 <= v <= MAX_K),
        policy=pick("policy", lambda v: v in POLICIES),
        decision_id=pick("decision_id", lambda v: re.fullmatch(r"D\d{4}", str(v)) is not None
                         and in_request(v)),
        student_id=pick("student_id", lambda v: re.fullmatch(r"S\d{3}|F-[\w-]+", str(v)) is not None
                        and in_request(v)),
        appeal_reason=pick("appeal_reason", lambda v: isinstance(v, str) and v.strip() != ""),
        source="llm",
    )
    return plan


# -------------------------------------------------------------- agent ------

def plan(request: str, ctx: RunContext) -> Plan:
    """Produce the plan and emit it to the trace."""
    rules = parse_rules(request)
    result = rules

    raw = ctx.ask_llm(NAME, "parse request", PROMPT.format(
        intents=list(INTENTS), domains=DOMAINS, areas=list(BANGALORE_AREAS),
        policies=list(POLICIES), max_k=MAX_K, k=DEFAULT_K, request=request), json_mode=True)
    if isinstance(raw, dict):
        result = _validate_llm(raw, rules, request)
        for key in ("decision_id", "student_id"):
            if raw.get(key) and raw[key] != getattr(result, key):
                ctx.trace.note(NAME, "LLM value ignored",
                               f"{key}={raw[key]} is not in the request; not using it.")
    elif raw is not None:
        ctx.trace.note(NAME, "LLM output ignored", "LLM did not return a JSON object; using rules.")

    if result.intent == "appeal" and not result.appeal_reason:
        result.appeal_reason = request.strip()
    result.steps = STEPS[result.intent]

    steps = " -> ".join(f"{agent}.{tool}" for agent, tool in result.steps)
    ctx.trace.note(NAME, f"plan ({result.source})",
                   f"intent={result.intent}; steps: {steps}", **result.params())
    return result
