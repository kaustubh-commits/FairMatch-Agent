"""
Planner: turns the user's request into a Plan (intent + parameters + steps).

Two ways to understand the request:
  - LLM mode: the LLM fills in a small JSON form. Every field is checked
    against the allowed values; anything invalid is replaced by the rule-based
    reading, so the LLM can never send the agents somewhere unexpected.
  - demo-safe mode: plain keyword/regex rules (`parse_rules`).

A structured form (the For Students page) can also hand the Planner a ready
profile; then no parsing is needed at all.

The STEPS for each intent are fixed below. The LLM only chooses which
intent and which parameters, never which tools exist. If a request matches
no intent, the plan is "help": the Planner says what it can do instead of
guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from agents.context import RunContext
from data_pipeline.constants import BANGALORE_AREAS, DOMAINS, KEYWORD_TO_SKILL, SKILLS
from tools.allocation import POLICIES
from tools.recommend import DEFAULT_HOURS

NAME = "Planner"
INTENTS = ("rank", "compare", "explain", "appeal", "recommend", "help")
DEFAULT_K = 3
MAX_K = 10

# Fixed tool-call recipes per intent: (agent, tool).
STEPS = {
    "rank": [("Matcher", "find_opportunities_near"), ("Matcher", "rank_candidates"),
             ("Matcher", "make_decision"), ("Equity Analyst", "fairness_metrics"),
             ("Auditor", "score_match"), ("Auditor", "log_decision"),
             ("Explainer", "explain record")],
    "compare": [("Matcher", "allocate"), ("Matcher", "simulate_rounds"),
                ("Equity Analyst", "fairness_metrics"), ("Auditor", "check allocation rules"),
                ("Explainer", "metrics table")],
    "explain": [("Auditor", "get_decision"), ("Explainer", "explain record")],
    "appeal": [("Auditor", "appeal_decision"), ("Explainer", "explain record")],
    "recommend": [("Matcher", "recommend_for_profile"), ("Matcher", "skills_to_learn"),
                  ("Explainer", "explain matches")],
    "help": [],
}

CAPABILITIES = """I can help with five kinds of request:
1. **Rank candidates**: "Find the best 3 candidates for the ML internships in Koramangala"
2. **Recommend internships for a student**: "I know Python and SQL, what can I do?" (or use the For Students page)
3. **Compare allocation policies**: "Compare the allocation policies"
4. **Explain a decision**: "Explain decision D0002" or "Explain the most recent decision"
5. **Appeal a decision**: "Appeal D0002 for S011: I have had no internships yet\""""

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

# Cues for each intent (whole words / phrases).
RANK_CUES = ["candidate", "candidates", "rank", "shortlist", "best", "top", "who should", "select", "pick"]
COMPARE_CUES = ["compare", "policy", "policies", "policy lab", "fairest", "fairness"]
RECOMMEND_CUES = ["i know", "i have", "i can", "i am", "i'm", "my skills", "what can i", "for me",
                  "recommend", "suggest", "what should i learn", "skills to learn", "suits me", "fit me"]

# Skill words: the skill names themselves plus the job-title keyword map.
SKILL_WORDS = {**KEYWORD_TO_SKILL, **{s.lower(): s for s in SKILLS},
               "web development": "Web dev", "app development": "App dev", "devops": "Cloud/DevOps"}
LEVEL_WORDS = {3: ["advanced", "expert", "strong", "very good", "proficient"],
               2: ["intermediate", "good", "decent", "comfortable"],
               1: ["beginner", "basic", "basics", "some", "a little", "learning", "new to"]}
DEFAULT_LEVEL = 2   # "I know Python" = intermediate


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
    skills: dict = field(default_factory=dict)   # recommend: {skill: level 1-3}
    hours: Optional[int] = None                  # recommend: hours per week
    steps: list = field(default_factory=list)
    source: str = "rules"            # "rules", "llm" or "form"

    def params(self) -> dict:
        """The non-empty parameters, for the trace."""
        keys = ("domain", "area", "k", "policy", "decision_id", "student_id", "appeal_reason",
                "skills", "hours")
        return {k: getattr(self, k) for k in keys if getattr(self, k) not in (None, "", {})}


# ------------------------------------------------------------- rules -------

def _has_word(text: str, word: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"(?!\w)", text) is not None


def _any(text: str, words: list[str]) -> bool:
    return any(_has_word(text, w) for w in words)


def parse_skills(text: str) -> dict[str, int]:
    """
    'I know advanced python and basic sql' -> {'Python': 3, 'Data analysis': 1}.
    The text is split into clauses (commas, 'and', 'but'); a level word in the
    same clause as a skill sets its level, otherwise it is intermediate (2).
    """
    skills: dict[str, int] = {}
    for clause in re.split(r",|;|\band\b|\bbut\b|\bplus\b", text.lower()):
        level = next((lvl for lvl, words in LEVEL_WORDS.items() if _any(clause, words)), DEFAULT_LEVEL)
        for word, skill in SKILL_WORDS.items():
            if _has_word(clause, word):
                skills[skill] = max(skills.get(skill, 0), level)
    return {s: skills[s] for s in SKILLS if s in skills}    # stable order


def parse_rules(request: str) -> Plan:
    """Rule-based reading of the request (used in demo-safe mode and to check the LLM)."""
    text = request.lower()

    decision = re.search(r"\bd\d{4}\b", text)
    student = re.search(r"\b(s\d{3}|f-[\w-]+)\b", text)
    wants_candidates = _any(text, RANK_CUES)

    if _any(text, ["appeal", "contest"]):
        intent = "appeal"
    elif decision or _has_word(text, "why was") or (_has_word(text, "explain") and _has_word(text, "decision")):
        intent = "explain"
    elif _any(text, RECOMMEND_CUES) and not wants_candidates:
        intent = "recommend"
    elif _any(text, COMPARE_CUES) and not wants_candidates:
        intent = "compare"
    elif wants_candidates:
        intent = "rank"
    else:
        intent = "help"            # don't guess: say what we can do

    domain = next((d for d, words in DOMAIN_WORDS if _any(text, words)), None)
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

    skills, hours = {}, None
    if intent == "recommend":
        skills = parse_skills(text)
        hours_match = re.search(r"(\d+)\s*(?:hours|hrs|h)\b", text)
        hours = int(hours_match.group(1)) if hours_match else None

    return Plan(intent=intent, domain=domain, area=area, k=max(1, min(k, MAX_K)),
                policy=policy, decision_id=decision.group().upper() if decision else None,
                student_id=_format_student(student.group()) if student else None,
                appeal_reason=reason, skills=skills, hours=hours)


def _format_student(raw: str) -> str:
    return raw.upper() if raw.lower().startswith("s") else "F-" + raw[2:]


# --------------------------------------------------------------- LLM -------

PROMPT = """You convert a request to an internship-matching system into JSON.
Do NOT answer the request, do NOT invent scores or reasons. Only fill this form.

Return exactly one JSON object with these keys:
  "intent":        one of {intents}
                   rank = find/rank candidates for opportunities (an allocator asking);
                   recommend = a student asking which internships fit their own skills;
                   compare = compare allocation policies / fairness;
                   explain = explain an existing decision;
                   appeal = a student appeals a decision;
                   help = none of the above
  "domain":        one of {domains} or null
  "area":          one of {areas} or null
  "k":             integer 1-{max_k} (how many candidates), default {k}
  "policy":        one of {policies} or null
  "decision_id":   like "D0001" ONLY if written in the request, else null
  "student_id":    like "S001" ONLY if written in the request, else null
  "appeal_reason": the student's reason in their own words, or null
  "skills":        for recommend: {{skill: level}} using only these skill names {skills},
                   level 1 = beginner, 2 = intermediate (default), 3 = advanced; else {{}}
  "hours":         hours per week if stated, else null

Request: {request}"""


def _skill_mentioned(skill: str, request: str) -> bool:
    """True if the request contains the skill's name or one of its keywords."""
    text = request.lower()
    return any(_has_word(text, word) for word, s in SKILL_WORDS.items() if s == skill)


def _validate_llm(raw: dict, fallback: Plan, request: str = "") -> Plan:
    """
    Keep each LLM field only if it is valid; otherwise use the rule-based value.
    Ids (D0001, S011) must also appear word for word in the request: an LLM once
    "filled in" D0001 for "explain the most recent decision", which is a made-up id.
    Skills must likewise be mentioned in the request.
    """
    def in_request(value) -> bool:
        return str(value).lower() in request.lower()

    def pick(key, ok):
        value = raw.get(key)
        return value if value is not None and ok(value) else getattr(fallback, key)

    areas_lower = {a.lower(): a for a in BANGALORE_AREAS}
    area = raw.get("area")
    area = areas_lower.get(str(area).lower()) if area else None

    skills = raw.get("skills") if isinstance(raw.get("skills"), dict) else {}
    skills = {s: lvl for s, lvl in skills.items()
              if s in SKILLS and isinstance(lvl, int) and 1 <= lvl <= 3 and _skill_mentioned(s, request)}

    return Plan(
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
        skills=skills or fallback.skills,
        hours=pick("hours", lambda v: isinstance(v, int) and 1 <= v <= 60 and in_request(v)),
        source="llm",
    )


# -------------------------------------------------------------- agent ------

def plan(request: str, ctx: RunContext, profile: Optional[dict] = None) -> Plan:
    """
    Produce the plan and emit it to the trace.
    `profile` ({"skills": {...}, "hours": int} or {"student_id": ...}) comes from a
    structured form; it is already unambiguous, so no parsing (and no LLM call) is needed.
    """
    if profile is not None:
        result = Plan(intent="recommend", skills=dict(profile.get("skills") or {}),
                      student_id=profile.get("student_id"), hours=profile.get("hours"),
                      k=profile.get("k", 5), source="form")
    else:
        rules = parse_rules(request)
        result = rules
        raw = ctx.ask_llm(NAME, "parse request", PROMPT.format(
            intents=list(INTENTS), domains=DOMAINS, areas=list(BANGALORE_AREAS),
            policies=list(POLICIES), max_k=MAX_K, k=DEFAULT_K, skills=SKILLS, request=request),
            json_mode=True)
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
    if result.intent == "recommend":
        result.hours = result.hours or DEFAULT_HOURS
        if result.k == DEFAULT_K and result.source != "form":
            result.k = 5
    result.steps = STEPS[result.intent]

    if result.intent == "help":
        ctx.trace.note(NAME, f"plan ({result.source})",
                       "Request doesn't match anything I can do; replying with my capabilities.")
    else:
        steps = " -> ".join(f"{agent}.{tool}" for agent, tool in result.steps)
        ctx.trace.note(NAME, f"plan ({result.source})",
                       f"intent={result.intent}; steps: {steps}", **result.params())
    return result
