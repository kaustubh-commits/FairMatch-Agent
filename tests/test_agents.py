"""
Agent tests. No network: the LLM is either absent (demo-safe) or a fake
client defined here, so we can make it fail or misbehave on purpose.
"""

import json

import pytest

from agents.grounding import ungrounded
from agents.orchestrator import AgentResult, run, stream
from agents.planner import Plan, _validate_llm, parse_rules, parse_skills
from llm import LLMClient, LLMError, make_client
from llm.client import ChainClient, describe, headline

DEMO_QUERY = "Find the best 3 candidates for the ML internships in Koramangala and explain why."


class FakeLLM(LLMClient):
    """Returns canned replies in order; raises LLMError when a reply is an Exception."""

    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, json_mode=False):
        self.prompts.append(prompt)
        reply = self.replies.pop(0) if self.replies else LLMError("no more replies")
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def path(tmp_path):
    return tmp_path / "decisions.json"


# ---- Planner ---------------------------------------------------------------

def test_rules_rank_demo_query():
    plan = parse_rules(DEMO_QUERY)
    assert (plan.intent, plan.domain, plan.area, plan.k) == ("rank", "ML", "Koramangala", 3)


@pytest.mark.parametrize("request_text, intent", [
    ("Compare the allocation policies", "compare"),
    ("Which policy is fairest?", "compare"),
    ("Explain decision D0002", "explain"),
    ("Explain the most recent decision", "explain"),
    ("Appeal the latest decision for the runner-up: no internships yet", "appeal"),
    ("Appeal D0001 for S011: I have more experience", "appeal"),
    ("Best 5 web candidates using the hybrid policy", "rank"),
    ("I know python and sql, what can I do?", "recommend"),
    ("Recommend internships for S011", "recommend"),
    ("What's the weather like?", "help"),
    ("Show me something", "help"),
])
def test_rules_intents(request_text, intent):
    assert parse_rules(request_text).intent == intent


def test_rules_appeal_fields():
    plan = parse_rules("Appeal D0001 for S011: I have more experience")
    assert (plan.decision_id, plan.student_id, plan.appeal_reason) == \
        ("D0001", "S011", "I have more experience")


def test_rules_number_words_and_policy():
    plan = parse_rules("top five design candidates in Whitefield, equal access please")
    assert (plan.k, plan.domain, plan.area, plan.policy) == (5, "Design", "Whitefield", "equal_access")


def test_llm_plan_invalid_fields_fall_back_to_rules():
    rules = parse_rules(DEMO_QUERY)
    raw = {"intent": "hack", "domain": "Rocket science", "area": "koramangala",
           "k": 500, "policy": "random", "decision_id": "DROP TABLE"}
    plan = _validate_llm(raw, rules)
    assert plan.intent == "rank" and plan.domain == "ML" and plan.k == 3
    assert plan.area == "Koramangala" and plan.policy is None and plan.decision_id is None


def test_llm_plan_cannot_invent_ids():
    request = "Explain the most recent decision."
    plan = _validate_llm({"intent": "explain", "decision_id": "D0001", "student_id": "S011"},
                         parse_rules(request), request)
    assert plan.intent == "explain" and plan.decision_id is None and plan.student_id is None


def test_llm_plan_keeps_ids_from_the_request():
    request = "Appeal D0002 for S011: no internships yet"
    plan = _validate_llm({"intent": "appeal", "decision_id": "D0002", "student_id": "S011"},
                         parse_rules(request), request)
    assert (plan.decision_id, plan.student_id) == ("D0002", "S011")


# ---- Grounding -------------------------------------------------------------

def test_grounding_accepts_rounded_and_percent():
    facts = {"skill_first": {"avg_fit": 0.685, "repeat_selection_rate": 0.432}}
    assert ungrounded("Skill-first fits best (0.69) but 43.2% are repeats.", facts) == []


def test_grounding_accepts_exact_four_decimals():
    assert ungrounded("Gini is 0.6078.", {"gini": 0.6078}) == []


def test_grounding_rejects_invented_numbers_and_ids():
    facts = {"selected": ["S025"], "score": 0.94}
    assert ungrounded("S025 scored 0.97 and beat S999.", facts) == ["S999", "0.97"]


# ---- Orchestrator: the four demo requests in demo-safe mode -----------------

def test_rank_flow_logs_decisions_and_runs_all_agents(path):
    result = run(DEMO_QUERY, client=None, decisions_path=path)
    assert result.intent == "rank" and result.decision_ids
    agents = [e.agent for e in result.events]
    for name in ("Planner", "Matcher", "Equity Analyst", "Auditor", "Explainer"):
        assert name in agents
    # Agents appear in the PLAN.md order.
    first = {name: agents.index(name) for name in set(agents)}
    assert first["Planner"] < first["Matcher"] < first["Equity Analyst"] < first["Auditor"] < first["Explainer"]
    assert result.mode_start == result.mode_end == "demo-safe"


def test_compare_flow(path):
    result = run("Compare the allocation policies", client=None, decisions_path=path)
    assert set(result.data["metrics"]) == {"skill_first", "equal_access", "hybrid"}
    assert "Takeaway (template)" in result.answer


def test_explain_and_appeal_flow(path):
    ranked = run(DEMO_QUERY, client=None, decisions_path=path)
    did = ranked.decision_ids[0]

    explained = run(f"Explain decision {did}", client=None, decisions_path=path)
    assert explained.intent == "explain" and did in explained.answer

    appealed = run(f"Appeal {did}: I have had no internships yet", client=None, decisions_path=path)
    assert appealed.intent == "appeal"
    assert appealed.data["record"]["appeal"]["outcome"] in ("upheld", "revised")


def test_explain_without_decisions_is_graceful(path):
    result = run("Explain decision D0042", client=None, decisions_path=path)
    assert "No such decision" in result.answer


# ---- LLM behaviour -----------------------------------------------------------

def test_llm_failure_mid_run_falls_back_to_demo_safe(path):
    # The planner call succeeds, then the API dies on the next call.
    plan_json = json.dumps({"intent": "rank", "domain": "ML", "area": "Koramangala", "k": 3})
    llm = FakeLLM([plan_json, LLMError("HTTP 503: overloaded")])
    result = run(DEMO_QUERY, client=llm, decisions_path=path)

    assert result.mode_start == "fake" and result.mode_end == "demo-safe"
    fallbacks = [e for e in result.events if e.action == "fallback to demo-safe"]
    assert len(fallbacks) == 1
    assert result.decision_ids                 # the run still finished
    assert len(llm.prompts) == 2               # no more LLM calls after the failure


def test_llm_invented_numbers_are_rejected(path):
    plan_json = json.dumps({"intent": "rank", "domain": "ML", "area": "Koramangala", "k": 3})
    lies = "S025 won with a perfect 0.99 score."
    llm = FakeLLM([plan_json] + [lies] * 5)
    result = run(DEMO_QUERY, client=llm, decisions_path=path)
    assert "0.99" not in result.answer
    assert any(e.action == "LLM summary rejected" for e in result.events)


def test_invented_id_is_ignored_and_traced(path):
    run(DEMO_QUERY, client=None, decisions_path=path)          # logs D0001, D0002
    llm = FakeLLM([json.dumps({"intent": "explain", "decision_id": "D0001"}), "Decision D0002."])
    result = run("Explain the most recent decision.", client=llm, decisions_path=path)
    assert result.decision_ids == ["D0002"]
    assert any(e.action == "LLM value ignored" for e in result.events)


def test_llm_grounded_summary_is_used(path):
    run(DEMO_QUERY, client=None, decisions_path=path)
    plan_json = json.dumps({"intent": "explain", "decision_id": "D0001"})
    # A summary containing only the decision id: always grounded.
    llm = FakeLLM([plan_json, "Decision D0001 went to the strongest candidate."])
    result = run("Explain decision D0001", client=llm, decisions_path=path)
    assert "Summary: Decision D0001 went to the strongest candidate." in result.answer


def test_stream_yields_events_then_result(path):
    items = list(stream("Compare the allocation policies", client=None, decisions_path=path))
    assert isinstance(items[-1], AgentResult)
    assert all(not isinstance(i, AgentResult) for i in items[:-1])
    assert items[0].agent == "System"


def test_make_client_demo_safe():
    client, note = make_client("demo-safe")
    assert client is None and "demo-safe" in note


# ---- Recommend + help -----------------------------------------------------------

def test_parse_skills_levels_and_keywords():
    assert parse_rules("I know python and sql, what can I do?").skills == {"Python": 2, "Data analysis": 2}
    assert parse_skills("advanced python, basic react and some figma design") == \
        {"Python": 3, "Web dev": 1, "Design": 1}


def test_llm_skills_must_be_mentioned():
    request = "I know python, what can I do?"
    plan = _validate_llm({"intent": "recommend", "skills": {"Python": 3, "ML": 3, "Cobol": 2}},
                         parse_rules(request), request)
    assert plan.skills == {"Python": 3}


def test_recommend_flow_uses_three_agents(path):
    result = run("I know python and sql, what can I do?", client=None, decisions_path=path)
    assert result.intent == "recommend"
    rec = result.data["recommend"]
    assert len(rec["matches"]) == 5 and rec["learn"]
    agents = {e.agent for e in result.events}
    assert {"Planner", "Matcher", "Explainer"} <= agents
    assert not {"Equity Analyst", "Auditor"} & agents
    assert result.decision_ids == []          # recommending never logs a decision


def test_recommend_without_skills_asks_for_them(path):
    result = run("What can I do? Recommend something for me", client=None, decisions_path=path)
    assert result.intent == "recommend" and "Tell me which skills" in result.answer


def test_form_profile_skips_parsing_and_llm(path):
    llm = FakeLLM(["A short summary."])
    result = run("form", client=llm, decisions_path=path, profile={"skills": {"ML": 3, "Python": 3}, "hours": 12})
    assert result.intent == "recommend" and result.data["plan"]["hours"] == 12
    assert len(llm.prompts) == 1               # only the Explainer; the Planner didn't call the LLM
    assert any(e.action == "plan (form)" for e in result.events)


def test_help_lists_capabilities(path):
    result = run("What's the weather like?", client=None, decisions_path=path)
    assert result.intent == "help" and "Rank candidates" in result.answer
    assert {e.agent for e in result.events} == {"System", "Planner"}


def test_compare_includes_simulation(path):
    result = run("Compare the allocation policies", client=None, decisions_path=path)
    sims = result.data["simulation"]
    assert set(sims) == {"skill_first", "equal_access", "hybrid"} and len(sims["hybrid"]) == 4
    assert result.data["rounds_takeaway"].startswith("After 4 rounds")


# ---- Friendly LLM errors + model fallback ------------------------------------------

class QuotaLLM(LLMClient):
    def __init__(self, name):
        self.name, self.calls = name, 0

    def complete(self, prompt, json_mode=False):
        self.calls += 1
        raise LLMError('HTTP 429: {"error": {"code": 429, "message": "quota"}}', "quota", self.name)


class OkLLM(LLMClient):
    def __init__(self, name, reply):
        self.name, self.reply, self.calls = name, reply, 0

    def complete(self, prompt, json_mode=False):
        self.calls += 1
        return self.reply


def test_friendly_error_text():
    err = LLMError("HTTP 429: {...}", "quota", "gemini:gemini-3.5-flash")
    assert describe(err) == "gemini-3.5-flash: quota reached"
    assert headline(err) == "Gemini quota reached"


def test_quota_falls_back_to_lite_model_then_skips_dead_model(path):
    flash = QuotaLLM("gemini:gemini-3.5-flash")
    lite = OkLLM("gemini:gemini-3.5-flash-lite", json.dumps({"intent": "help"}))
    chain = ChainClient([flash, lite])
    result = run("hello there", client=chain, decisions_path=path)
    notes = [e for e in result.events if e.action == "LLM fallback"]
    assert notes and "quota reached" in notes[0].result_summary and "flash-lite" in notes[0].result_summary
    assert "429" in notes[0].args["error_detail"]           # raw details kept for the expander
    chain.complete("again")
    assert flash.calls == 1 and lite.calls == 2             # flash is not retried after a quota error


def test_total_failure_shows_friendly_line_and_details(path):
    chain = ChainClient([QuotaLLM("gemini:gemini-3.5-flash"), QuotaLLM("gemini:gemini-3.5-flash-lite")])
    result = run(DEMO_QUERY, client=chain, decisions_path=path)
    failed = next(e for e in result.events if e.action == "LLM: parse request")
    assert failed.result_summary == "Failed: gemini-3.5-flash: quota reached · gemini-3.5-flash-lite: quota reached"
    assert "{" not in failed.result_summary and "429" in failed.args["error_detail"]
    fallback = next(e for e in result.events if e.action == "fallback to demo-safe")
    assert fallback.result_summary == "Gemini quota reached, continuing rule-based (demo-safe)."
    assert result.decision_ids                              # the run still finished
