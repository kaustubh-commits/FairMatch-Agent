"""
Orchestrator: runs Planner -> Matcher -> Equity Analyst -> Auditor -> Explainer.

Two ways to use it:

    # 1. Callback: on_event(event) is called the moment each step finishes.
    result = run("Find the best 3 ML candidates in Koramangala", on_event=print)

    # 2. Generator: iterate live. Yields StepStarted when a step begins,
    #    TraceEvent when it ends, a PartialResult as soon as the tables are
    #    ready (before the Explainer writes), and finally the AgentResult.
    for item in stream("Compare the policies"):
        ...

Which agents run depends on the plan's intent:
  rank       Planner, Matcher, Equity Analyst, Auditor, Explainer
  compare    Planner, Matcher, Equity Analyst, Auditor, Explainer
  recommend  Planner, Matcher, Explainer
  explain    Planner, Auditor, Explainer
  appeal     Planner, Auditor, Explainer
  help       Planner only (the request matched nothing; it lists what it can do)

Two orchestrators do this routing, chosen by ORCHESTRATOR in .env:
  langgraph  (default) agents/graph.py: a LangGraph StateGraph with a human-approval
             pause and a SQLite checkpointer
  classic    the plain Python handlers at the bottom of this file
Both call the SAME agent functions and the SAME formatting helpers below, so their
decisions, scores and answers are identical (tests/test_graph.py checks this).
If LangGraph can't be imported, the run falls back to classic with a trace note.
"""

from __future__ import annotations

import importlib
import os
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

from agents import auditor, equity, explainer, matcher, planner
from agents.context import RunContext
from agents.trace import PartialResult, StepStarted, Trace, TraceEvent
from llm import make_client
from llm.client import load_env
from tools.decisions import DEFAULT_PATH
from tools.store import get_opportunity

_UNSET = object()

ORCHESTRATORS = {"langgraph": "LangGraph", "classic": "Classic"}   # value in .env -> display name
DEFAULT_ORCHESTRATOR = "langgraph"


@dataclass
class AgentResult:
    request: str
    intent: str
    answer: str                      # text shown to the user
    mode_start: str                  # LLM at the start of the run (or demo-safe)
    mode_end: str                    # LLM at the end (demo-safe if it fell back)
    decision_ids: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)   # structured results for the UI
    events: list[TraceEvent] = field(default_factory=list)


# ------------------------------------------------------ which orchestrator --

def chosen_orchestrator() -> str:
    """ORCHESTRATOR from the environment / .env: 'langgraph' (default) or 'classic'."""
    load_env()
    return (os.environ.get("ORCHESTRATOR") or DEFAULT_ORCHESTRATOR).strip().lower()


def pick_orchestrator(orchestrator: Optional[str]):
    """
    Returns (graph module or None, problem note or None).
    None as the module means: use classic. The note says why, if it's a fallback.
    """
    name = (orchestrator or chosen_orchestrator()).lower()
    if name == "classic":
        return None, None
    if name != "langgraph":
        return None, f"ORCHESTRATOR={name} is not langgraph or classic; using classic."
    try:
        return importlib.import_module("agents.graph"), None
    except ImportError as e:      # langgraph or the SQLite checkpointer not installed
        return None, f"LangGraph could not be imported ({e}); using the classic orchestrator."


# --------------------------------------------------- shared start / finish --

def start_run(request: str, provider: Optional[str], client, trace: Trace,
              orchestrator: str, problem: Optional[str] = None) -> RunContext:
    """Create the LLM client and emit the 'System start' lines (same for both orchestrators)."""
    if client is _UNSET:
        client, note = make_client(provider)
    else:
        note = f"using injected client {client.name}" if client else "demo-safe mode"
    label = ORCHESTRATORS[orchestrator]
    trace.note("System", "start", f"orchestrator: {label}; {note}", request=request,
               orchestrator=orchestrator)
    if problem:
        trace.note("System", "orchestrator fallback", problem)
    # A plain line for the default view too, so the demo shows which orchestrator ran.
    trace.emit(TraceEvent(agent="System", action="orchestrator", kind="headline",
                          result_summary=f"Orchestrator: {label}"
                          + (" (fallback; see developer details for why)." if problem else ".")))
    return RunContext(client, trace)


def finish_run(request: str, plan, ctx: RunContext, mode_start: str,
               answer: str, data: dict, ids: list) -> AgentResult:
    mode = mode_start if ctx.mode == mode_start else f"{mode_start} -> {ctx.mode}"
    ctx.trace.note("System", "done", f"intent={plan.intent}; mode {mode}")
    return AgentResult(request=request, intent=plan.intent, answer=answer,
                       mode_start=mode_start, mode_end=ctx.mode, decision_ids=ids,
                       data={"plan": plan.params(), **data}, events=ctx.trace.events)


# ------------------------------------------------------------- run/stream --

def run(request: str, provider: Optional[str] = None,
        on_event: Optional[Callable[[TraceEvent], None]] = None,
        decisions_path: Path = DEFAULT_PATH,
        client=_UNSET,
        on_start: Optional[Callable[[StepStarted], None]] = None,
        profile: Optional[dict] = None,
        on_partial: Optional[Callable[[PartialResult], None]] = None,
        orchestrator: Optional[str] = None) -> AgentResult:
    """
    Handle one request end to end.
      provider        gemini | ollama | demo-safe (default: LLM_PROVIDER or gemini)
      client          pass an LLMClient (or None) directly; used by tests
      profile         a student profile from a form ({"skills": ..., "hours": ...} or
                      {"student_id": ...}); skips request parsing and runs "recommend"
      on_partial      called with a PartialResult once the tables are ready
      orchestrator    langgraph | classic (default: ORCHESTRATOR in .env, else langgraph)
    """
    graph, problem = pick_orchestrator(orchestrator)
    if graph is not None:
        # Same callbacks, fed from the graph's stream.
        for item in graph.stream_run(request, provider, decisions_path, client, profile):
            if isinstance(item, StepStarted):
                on_start and on_start(item)
            elif isinstance(item, TraceEvent):
                on_event and on_event(item)
            elif isinstance(item, PartialResult):
                on_partial and on_partial(item)
            else:
                return item
        raise RuntimeError("graph run ended without a result")
    return run_classic(request, provider, on_event, decisions_path, client, on_start,
                       profile, on_partial, problem)


def stream(request: str, provider: Optional[str] = None,
           decisions_path: Path = DEFAULT_PATH,
           client=_UNSET, profile: Optional[dict] = None,
           orchestrator: Optional[str] = None
           ) -> Iterator[Union[StepStarted, TraceEvent, PartialResult, AgentResult]]:
    """
    Generator version of run(): yields a StepStarted when each step begins,
    its TraceEvent when it ends, and the AgentResult last.
    LangGraph streams these itself; classic runs in a background thread so events
    arrive while it is still going.
    """
    graph, problem = pick_orchestrator(orchestrator)
    if graph is not None:
        yield from graph.stream_run(request, provider, decisions_path, client, profile)
        return

    events: queue.Queue = queue.Queue()
    done = object()
    outcome = {}

    def worker():
        try:
            outcome["result"] = run_classic(request, provider, events.put, decisions_path, client,
                                            events.put, profile, events.put, problem)
        except Exception as e:      # re-raised in the caller's thread below
            outcome["error"] = e
        events.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while (item := events.get()) is not done:
        yield item
    if "error" in outcome:
        raise outcome["error"]
    yield outcome["result"]


# ------------------------------------------------- classic orchestrator ----

def run_classic(request, provider=None, on_event=None, decisions_path: Path = DEFAULT_PATH,
                client=_UNSET, on_start=None, profile=None, on_partial=None,
                problem: Optional[str] = None) -> AgentResult:
    """The original orchestrator: one Python handler per intent."""
    trace = Trace(on_event, on_start)
    ctx = start_run(request, provider, client, trace, "classic", problem)
    ctx.on_partial = on_partial or (lambda partial: None)
    mode_start = ctx.mode

    plan = planner.plan(request, ctx, profile)
    handler = {"rank": _rank, "compare": _compare, "recommend": _recommend,
               "explain": _explain, "appeal": _appeal, "help": _help}[plan.intent]
    answer, data, ids = handler(plan, ctx, decisions_path)
    return finish_run(request, plan, ctx, mode_start, answer, data, ids)


def _rank(plan, ctx, path):
    match = matcher.rank(plan, ctx)
    if not match["found"]["opportunities"]:
        return no_matches(match["found"])
    check = equity.check_proposals(match, ctx)
    logged = auditor.verify_and_log(match["proposals"], path, ctx)
    data, ids = rank_tables(plan, match, check, logged)
    ctx.on_partial(PartialResult("rank", dict(data), ids))      # tables can be shown now
    summaries, texts = explainer.explain_all(logged, ctx)          # ONE LLM call for all decisions
    return format_rank(plan, match, check, logged, data, summaries, texts)


def _compare(plan, ctx, path):
    allocations, sims = matcher.compare(plan, ctx)
    result = equity.compare_policies(allocations, sims, ctx)
    audit = auditor.check_allocations(allocations, ctx)
    return format_compare(plan, result, sims, audit, ctx)


def _recommend(plan, ctx, path):
    if not has_profile(plan):
        return missing_skills(ctx)
    result = matcher.recommend(plan, ctx)
    ctx.on_partial(PartialResult("recommend", {"recommend": result}))   # show matches now
    summary = explainer.explain_matches(result, ctx)
    return format_recommend(result, summary)


def _help(plan, ctx, path):
    return planner.CAPABILITIES, {}, []


def _explain(plan, ctx, path):
    record = auditor.fetch(plan.decision_id, path, ctx)
    if record is None:
        return no_record("explain")
    return format_record(record, *explainer.explain(record, ctx))


def _appeal(plan, ctx, path):
    record = auditor.appeal(plan.decision_id, plan.appeal_reason, plan.student_id, path, ctx)
    if record is None:
        return no_record("appeal")
    return format_record(record, *explainer.explain(record, ctx))


# ------------------------------------------- shared formatting helpers ----
# Used by both orchestrators. Each returns (answer, data, decision_ids) unless noted.

def has_profile(plan) -> bool:
    """A recommend request needs skills or a student id."""
    return bool(plan.skills or plan.student_id)


def no_matches(found: dict):
    return found["note"] or "No matching opportunities found.", {"match": None}, []


def no_record(intent: str):
    if intent == "appeal":
        return ("No such decision to appeal. Run a ranking request first, or name a decision like D0001.",
                {}, [])
    return "No such decision. Run a ranking request first, or name a decision like D0001.", {}, []


def missing_skills(ctx: RunContext):
    ctx.trace.note("Planner", "missing skills", "No skills or student id found in the request.")
    ctx.trace.headline("Planner", "I need to know the student's skills first.")
    return ("Tell me which skills you have (for example: \"I know advanced Python and basic SQL\") "
            "or give a student id, and I'll find matching internships. "
            "The For Students page lets you pick skills from a list.", {"recommend": None}, [])


def rank_tables(plan, match: dict, check: dict, logged: list) -> tuple[dict, list]:
    """The rank tables (shown before the Explainer writes) and the logged decision ids."""
    found = match["found"]
    data = {"found_note": found["note"], "area_used": found["area_used"], "policy": match["policy"],
            "max_km": plan.max_km, "rankings": match["rankings"],
            "equity": {k: v for k, v in check.items() if k != "alternatives"},
            "alternatives": {k: v.selected for k, v in check["alternatives"].items()}}
    return data, [r.id for r in logged]


def format_rank(plan, match: dict, check: dict, logged: list, data: dict,
                summaries: dict, texts: dict):
    found = match["found"]
    lines = [f"Note: {found['note']}"] if found["note"] else []
    for record in logged:
        opp = get_opportunity(record.opportunity_id)
        lines.append(f"\n{opp.id}: {opp.title} @ {opp.company} ({opp.area}), {opp.slots} slot(s)")
        lines.append(f"Top {plan.k} candidates by match score:")
        for i, m in enumerate(match["rankings"][opp.id], start=1):
            skills = ", ".join(f"{s} {b['level']}/3 ({b['kind'][:3]})" for s, b in m["breakdown"].items())
            commute = f", {m['commute_km']} km" if m["commute_km"] is not None else ""
            lines.append(f"  {i}. {m['student_label']}  match {m['score']:.2f} (skill fit {m['skill_fit']:.2f}"
                         f"{commute})  past opportunities {m['past_opportunities']}  | {skills}")
        lines.append(texts[record.id])
    lines.append("\n" + check["message"])
    return "\n".join(lines).strip(), {**data, "summaries": summaries}, [r.id for r in logged]


def format_compare(plan, result: dict, sims: dict, audit: dict, ctx: RunContext):
    with ctx.trace.step("Explainer", "format comparison", tool="metrics table") as ev:
        header = (f"{'policy':<14}{'skill fit':>10}{'gini':>8}{'unique':>8}{'repeat %':>10}"
                  f"{'commute':>9}{'outskirts gap':>15}")
        lines = [header, "-" * len(header)]
        for policy, m in result["table"].items():
            lines.append(f"{policy:<14}{m['avg_fit']:>10.3f}{m['gini']:>8.3f}{m['unique_recipients']:>8}"
                         f"{m['repeat_selection_rate'] * 100:>9.1f}%{m['avg_commute_km']:>6} km"
                         f"{m['outskirts_gap_pts']:>11} pts")
        lines.append(f"\nTakeaway ({result['takeaway_source']}): {result['takeaway']}")
        lines.append(f"Over {len(sims['hybrid'])} rounds ({result['rounds_takeaway_source']}): "
                     f"{result['rounds_takeaway']}")
        lines.append(result["outskirts"])
        ok = all(not (r["over_cap"] or r["over_slots"]) for r in audit.values())
        lines.append("Auditor: all allocations respect slots and the per-student cap."
                     if ok else f"Auditor found violations: {audit}")
        ev.result_summary = f"table of {len(result['table'])} policies"
    ctx.trace.headline("Explainer", "Put the three policies side by side with the Equity Analyst's takeaway.")
    return "\n".join(lines), {"metrics": result["table"], "takeaway": result["takeaway"],
                              "takeaway_source": result["takeaway_source"],
                              "simulation": sims, "rounds_takeaway": result["rounds_takeaway"],
                              "rounds_takeaway_source": result["rounds_takeaway_source"],
                              "outskirts": result["outskirts"], "outskirts_flags": result["outskirts_flags"],
                              "max_km": plan.max_km, "audit": audit}, []


def format_recommend(result: dict, summary: str):
    lines = [summary, ""]
    for i, m in enumerate(result["matches"], start=1):
        skills = ", ".join(f"{s} {b['level']}/3" for s, b in m["breakdown"].items())
        commute = f", {m['commute_km']} km away" if m["commute_km"] is not None else ""
        lines.append(f"{i}. {m['opportunity_id']} {m['title']} @ {m['company']} ({m['area']}) "
                     f"match {m['score']:.2f} (skill fit {m['skill_fit']:.2f}{commute}) | {skills}")
    learn = result["learn"]
    if learn["next"]:
        lines.append(f"\nSkills to learn next (in {', '.join(learn['interests'])}): " + ", ".join(
            f"{r['skill']} (+{r['unlocks_in_interests']})" for r in learn["next"]))
    if learn["also_opens"]:
        lines.append("Also opens, in other areas: " + ", ".join(
            f"{r['skill']} (+{r['unlocks_elsewhere']})" for r in learn["also_opens"]))
    return "\n".join(lines), {"recommend": result, "summary": summary}, []


def format_record(record, summary: str, text: str):
    return text, {"record": record.to_dict(), "summary": summary}, [record.id]
