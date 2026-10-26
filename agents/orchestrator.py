"""
Orchestrator: runs Planner -> Matcher -> Equity Analyst -> Auditor -> Explainer.

Two ways to use it:

    # 1. Callback: on_event(event) is called the moment each step finishes.
    result = run("Find the best 3 ML candidates in Koramangala", on_event=print)

    # 2. Generator: iterate over events live; the last item is the AgentResult.
    for item in stream("Compare the policies"):
        ...

Which agents run depends on the plan's intent:
  rank     Planner, Matcher, Equity Analyst, Auditor, Explainer
  compare  Planner, Matcher, Equity Analyst, Auditor, Explainer
  explain  Planner, Auditor, Explainer
  appeal   Planner, Auditor, Explainer
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Union

from agents import auditor, equity, explainer, matcher, planner
from agents.context import RunContext
from agents.trace import Trace, TraceEvent
from llm import LLMClient, make_client
from tools.decisions import DEFAULT_PATH
from tools.store import get_opportunity

_UNSET = object()


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


def run(request: str, provider: Optional[str] = None,
        on_event: Optional[Callable[[TraceEvent], None]] = None,
        decisions_path: Path = DEFAULT_PATH,
        client=_UNSET) -> AgentResult:
    """
    Handle one request end to end.
      provider        gemini | ollama | demo-safe (default: LLM_PROVIDER or gemini)
      client          pass an LLMClient (or None) directly; used by tests
    """
    trace = Trace(on_event)
    if client is _UNSET:
        client, note = make_client(provider)
    else:
        note = f"using injected client {client.name}" if client else "demo-safe mode"
    trace.note("System", "start", note, request=request)
    ctx = RunContext(client, trace)
    mode_start = ctx.mode

    plan = planner.plan(request, ctx)
    handler = {"rank": _rank, "compare": _compare,
               "explain": _explain, "appeal": _appeal}[plan.intent]
    answer, data, ids = handler(plan, ctx, decisions_path)

    mode = mode_start if ctx.mode == mode_start else f"{mode_start} -> {ctx.mode}"
    trace.note("System", "done", f"intent={plan.intent}; mode {mode}")
    return AgentResult(request=request, intent=plan.intent, answer=answer,
                       mode_start=mode_start, mode_end=ctx.mode, decision_ids=ids,
                       data={"plan": plan.params(), **data}, events=trace.events)


def stream(request: str, provider: Optional[str] = None,
           decisions_path: Path = DEFAULT_PATH,
           client=_UNSET) -> Iterator[Union[TraceEvent, AgentResult]]:
    """
    Generator version of run(): yields each TraceEvent as it happens, then
    the AgentResult last. The work runs in a background thread so events
    arrive while it is still going.
    """
    events: queue.Queue = queue.Queue()
    done = object()
    outcome = {}

    def worker():
        try:
            outcome["result"] = run(request, provider, on_event=events.put,
                                    decisions_path=decisions_path, client=client)
        except Exception as e:      # re-raised in the caller's thread below
            outcome["error"] = e
        events.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while (item := events.get()) is not done:
        yield item
    if "error" in outcome:
        raise outcome["error"]
    yield outcome["result"]


# ------------------------------------------------------------ intents -------

def _rank(plan, ctx, path):
    match = matcher.rank(plan, ctx)
    found = match["found"]
    if not found["opportunities"]:
        return found["note"] or "No matching opportunities found.", {"match": None}, []

    check = equity.check_proposals(match, ctx)
    logged = auditor.verify_and_log(match["proposals"], path, ctx)

    lines = []
    if found["note"]:
        lines.append(f"Note: {found['note']}")
    for record in logged:
        opp = get_opportunity(record.opportunity_id)
        lines.append(f"\n{opp.id}: {opp.title} @ {opp.company} ({opp.area}), {opp.slots} slot(s)")
        lines.append(f"Top {plan.k} candidates by skill fit:")
        for i, m in enumerate(match["rankings"][opp.id], start=1):
            skills = ", ".join(f"{s} {b['level']}/3 ({b['kind'][:3]})" for s, b in m["breakdown"].items())
            lines.append(f"  {i}. {m['student_id']}  fit {m['score']:.2f}  "
                         f"past {m['past_opportunities']}  | {skills}")
        lines.append(explainer.explain(record, ctx))
    lines.append("\n" + check["message"])

    data = {"found_note": found["note"], "area_used": found["area_used"],
            "rankings": match["rankings"], "equity": {k: v for k, v in check.items() if k != "alternatives"},
            "alternatives": {k: v.selected for k, v in check["alternatives"].items()}}
    return "\n".join(lines).strip(), data, [r.id for r in logged]


def _compare(plan, ctx, path):
    allocations = matcher.allocate_all(ctx)
    result = equity.compare_policies(allocations, ctx)
    audit = auditor.check_allocations(allocations, ctx)

    with ctx.trace.step("Explainer", "format comparison", tool="metrics table") as ev:
        header = f"{'policy':<14}{'avg fit':>9}{'gini':>8}{'unique':>9}{'repeat %':>10}{'slots':>7}"
        lines = [header, "-" * len(header)]
        for policy, m in result["table"].items():
            lines.append(f"{policy:<14}{m['avg_fit']:>9.3f}{m['gini']:>8.3f}{m['unique_recipients']:>9}"
                         f"{m['repeat_selection_rate'] * 100:>9.1f}%{m['slots_filled']:>7}")
        lines.append(f"\nTakeaway ({result['takeaway_source']}): {result['takeaway']}")
        ok = all(not (r["over_cap"] or r["over_slots"]) for r in audit.values())
        lines.append("Auditor: all allocations respect slots and the per-student cap."
                     if ok else f"Auditor found violations: {audit}")
        ev.result_summary = f"table of {len(result['table'])} policies"
    return "\n".join(lines), {"metrics": result["table"], "takeaway": result["takeaway"],
                              "audit": audit}, []


def _explain(plan, ctx, path):
    record = auditor.fetch(plan.decision_id, path, ctx)
    if record is None:
        return ("No such decision. Run a ranking request first, or name a decision like D0001.",
                {}, [])
    return explainer.explain(record, ctx), {"record": record.to_dict()}, [record.id]


def _appeal(plan, ctx, path):
    record = auditor.appeal(plan.decision_id, plan.appeal_reason, plan.student_id, path, ctx)
    if record is None:
        return ("No such decision to appeal. Run a ranking request first, or name a decision like D0001.",
                {}, [])
    return explainer.explain(record, ctx), {"record": record.to_dict()}, [record.id]
