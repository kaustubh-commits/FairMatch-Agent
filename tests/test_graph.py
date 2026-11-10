"""
LangGraph orchestrator tests. Demo-safe only (client=None), no network.

The key test: classic and LangGraph produce IDENTICAL decisions, scores, selections,
answers and trace lines for all six intents. Only the orchestrator's own "System"
lines (which name the orchestrator) are allowed to differ.
"""

import json
import sys

import pytest

from agents import graph
from agents.orchestrator import AgentResult, run, stream
from agents.planner import STEPS
from agents.trace import PartialResult, StepStarted, TraceEvent
from tools.decisions import get_decision, list_decisions

DEMO_QUERY = "Find the best 3 candidates for the ML internships in Koramangala and explain why."

# The six intents, in an order where explain/appeal have a decision to work on.
INTENT_REQUESTS = [
    ("rank", DEMO_QUERY),
    ("compare", "Compare the allocation policies"),
    ("recommend", "I know python and sql, what can I do?"),
    ("explain", "Explain decision D0001"),
    ("appeal", "Appeal D0001 for the runner-up: I have had no internships yet"),
    ("help", "What's the weather like?"),
]


def _comparable_events(result: AgentResult) -> list[tuple]:
    """Trace lines without timings and without the System lines that name the orchestrator."""
    return [(e.agent, e.action, e.tool, e.result_summary, e.kind)
            for e in result.events if not (e.agent == "System" and e.action in ("start", "orchestrator"))]


def _records(path) -> list[dict]:
    """Every logged decision, minus its timestamp."""
    return [{k: v for k, v in r.to_dict().items() if k != "timestamp"} for r in list_decisions(path)]


def _without_times(data: dict) -> dict:
    """A run's data with the decision record's timestamp removed (runs happen at different moments)."""
    if isinstance(data.get("record"), dict):
        data = {**data, "record": {k: v for k, v in data["record"].items() if k != "timestamp"}}
    return data


def _run_all(orchestrator: str, path) -> list[AgentResult]:
    return [run(text, client=None, decisions_path=path, orchestrator=orchestrator)
            for _, text in INTENT_REQUESTS]


def test_classic_and_langgraph_are_identical_for_all_six_intents(tmp_path):
    classic_path = tmp_path / "classic" / "decisions.json"
    graph_path = tmp_path / "graph" / "decisions.json"
    classic = _run_all("classic", classic_path)
    lg = _run_all("langgraph", graph_path)

    for (intent, _), c, g in zip(INTENT_REQUESTS, classic, lg):
        assert c.intent == g.intent == intent
        assert c.answer == g.answer, intent
        assert c.decision_ids == g.decision_ids, intent
        assert _without_times(c.data) == _without_times(g.data), intent   # scores, rankings, metrics
        assert _comparable_events(c) == _comparable_events(g), intent
    # The decision log itself: same records, selections and scores, and the
    # appeal changed the same decision in the same way.
    assert _records(classic_path) == _records(graph_path)
    assert len(_records(graph_path)) >= 1


def test_start_line_names_the_orchestrator(tmp_path):
    for name, label in (("langgraph", "LangGraph"), ("classic", "Classic")):
        result = run("Compare the allocation policies", client=None,
                     decisions_path=tmp_path / name / "d.json", orchestrator=name)
        start = result.events[0]
        assert (start.agent, start.action) == ("System", "start")
        assert start.result_summary.startswith(f"orchestrator: {label}")         # developer view
        headline = next(e for e in result.events if e.kind == "headline")
        assert headline.result_summary == f"Orchestrator: {label}."            # normal view


@pytest.mark.parametrize("intent, text", INTENT_REQUESTS)
def test_each_intent_takes_its_fixed_path(tmp_path, intent, text):
    path = tmp_path / "decisions.json"
    if intent in ("explain", "appeal"):
        run(DEMO_QUERY, client=None, decisions_path=path, orchestrator="langgraph")
    graph.run_graph(text, client=None, decisions_path=path, thread_id="t-" + intent)
    visited = graph.get_app(path).get_state({"configurable": {"thread_id": "t-" + intent}}).values["visited"]

    node_of = {"Matcher": "matcher", "Equity Analyst": "equity_analyst", "Auditor": "auditor",
               "Explainer": "explainer"}
    expected = ["planner"]
    for agent, _ in STEPS[intent]:
        if node_of[agent] != expected[-1]:
            expected.append(node_of[agent])
    if intent == "help":
        expected.append("respond")
    assert visited == expected
    # A rank run then pauses at human_approval, waiting for a person; the others end.
    snapshot = graph.get_app(path).get_state({"configurable": {"thread_id": "t-" + intent}})
    assert snapshot.next == (("human_approval",) if intent == "rank" else ())


def test_recommend_without_skills_goes_to_respond(tmp_path):
    graph.run_graph("recommend something for me", client=None,
                    decisions_path=tmp_path / "d.json", thread_id="t-noskills")
    state = graph.get_app(tmp_path / "d.json").get_state({"configurable": {"thread_id": "t-noskills"}})
    assert state.values["visited"] == ["planner", "respond"]


def test_rank_pauses_for_approval_and_resume_approves(tmp_path):
    path = tmp_path / "decisions.json"
    result = run(DEMO_QUERY, client=None, decisions_path=path, orchestrator="langgraph")
    ids = result.decision_ids
    assert len(ids) >= 2
    assert all(get_decision(i, path).status == "proposed" for i in ids)
    assert all(graph.is_paused(i, path) for i in ids)

    # Approve the first: the run resumes, approves it, and pauses again for the rest.
    assert graph.resume_approval(ids[0], path) is True
    assert get_decision(ids[0], path).status == "approved"
    assert not graph.is_paused(ids[0], path)
    assert graph.is_paused(ids[1], path)
    assert graph.resume_approval(ids[0], path) is False      # already done

    for i in ids[1:]:
        assert graph.resume_approval(i, path) is True
    assert all(get_decision(i, path).status == "approved" for i in ids)
    thread = json.loads((tmp_path / "pending_approvals.json").read_text())
    assert thread == {}                                       # the run has ended


def test_resume_unknown_or_classic_decision_returns_false(tmp_path):
    path = tmp_path / "decisions.json"
    result = run(DEMO_QUERY, client=None, decisions_path=path, orchestrator="classic")
    assert graph.resume_approval(result.decision_ids[0], path) is False    # caller approves directly
    assert graph.resume_approval("D9999", path) is False


def test_paused_run_survives_a_restart(tmp_path):
    """A fresh checkpointer on the same SQLite file (like an app restart) can resume the run."""
    path = tmp_path / "decisions.json"
    result = run(DEMO_QUERY, client=None, decisions_path=path, orchestrator="langgraph")
    graph._compiled.clear()                                   # forget the compiled graph
    assert graph.resume_approval(result.decision_ids[0], path) is True
    assert get_decision(result.decision_ids[0], path).status == "approved"


def test_langgraph_stream_shows_tables_before_explainer(tmp_path):
    items = list(stream(DEMO_QUERY, client=None, decisions_path=tmp_path / "d.json",
                        orchestrator="langgraph"))
    assert isinstance(items[-1], AgentResult)
    assert isinstance(items[0], TraceEvent) and items[0].agent == "System"
    kinds = [type(i).__name__ for i in items]
    partial_at = kinds.index("PartialResult")
    first_explainer = next(n for n, it in enumerate(items)
                           if isinstance(it, (TraceEvent, StepStarted)) and it.agent == "Explainer")
    assert partial_at < first_explainer
    assert isinstance(items[partial_at], PartialResult) and items[partial_at].decision_ids


def test_falls_back_to_classic_if_langgraph_cannot_be_imported(tmp_path, monkeypatch):
    # Pretend the SQLite checkpointer isn't installed, and force agents.graph to re-import.
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.sqlite", None)
    monkeypatch.delitem(sys.modules, "agents.graph")
    path = tmp_path / "decisions.json"
    result = run(DEMO_QUERY, client=None, decisions_path=path, orchestrator="langgraph")

    assert result.intent == "rank" and result.decision_ids            # the run still worked
    assert result.events[0].result_summary.startswith("orchestrator: Classic")
    fallback = [e for e in result.events if e.action == "orchestrator fallback"]
    assert len(fallback) == 1 and "LangGraph could not be imported" in fallback[0].result_summary
    headline = next(e for e in result.events if e.action == "orchestrator")
    assert "fallback" in headline.result_summary
    assert not (tmp_path / "checkpoints.sqlite").exists()              # graph never ran


def test_unknown_orchestrator_setting_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR", "spaghetti")
    result = run("Compare the allocation policies", client=None, decisions_path=tmp_path / "d.json")
    assert result.events[0].result_summary.startswith("orchestrator: Classic")
    assert any(e.action == "orchestrator fallback" for e in result.events)
