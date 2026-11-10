"""
The LangGraph orchestrator: the same five agents, wired as a StateGraph.

    planner ──rank/compare/recommend──> matcher ──rank/compare──> equity_analyst ──> auditor ──> explainer
       │                                   │                                           ▲            │
       ├──explain/appeal───────────────────┼───────────────────────────────────────────┘            │
       │                                   └──recommend──> explainer                                ▼
       └──help / nothing to do──> respond ──> END                       human_approval (rank only) ──> END

What LangGraph does here:
  - ROUTING: conditional edges send each intent along its fixed path
    (the same steps as planner.STEPS; the LLM never picks the path).
  - HUMAN APPROVAL: after a rank run, `human_approval` calls interrupt() and the
    run PAUSES until a person presses Approve in the Decision Log.
  - CHECKPOINTING: the paused run is saved in SQLite, so it can be resumed later,
    even after the app restarts.
What it does NOT do: decide anything. Every node just calls an existing agent
function; scores come from tools/, rules from the Auditor, approval from a human.

The trace still works the same way: agents write to Trace, and Trace forwards each
event into LangGraph's custom stream; stream_run() turns the stream back into the
usual StepStarted / TraceEvent / PartialResult objects, then the AgentResult.
"""

from __future__ import annotations

import json
import operator
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any, Iterator, Optional, TypedDict

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents import auditor, equity, explainer, matcher, planner
from agents.orchestrator import (_UNSET, finish_run, format_compare, format_rank, format_record,
                                 format_recommend, has_profile, missing_skills, no_matches, no_record,
                                 rank_tables, start_run)
from agents.trace import PartialResult, StepStarted, Trace, TraceEvent
from tools.decisions import DEFAULT_PATH, approve_decision, get_decision


# ------------------------------------------------------------------ state --

class State(TypedDict, total=False):
    """Everything the agents share during one run (saved at every step by the checkpointer)."""
    request: str
    profile: Optional[dict]
    plan: Any                     # planner.Plan
    match: dict                   # rank: found listings, rankings, proposals (DecisionRecords)
    allocations: dict             # compare: each policy's allocation
    sims: dict                    # compare: multi-round simulation
    equity: dict                  # equity flags: check_proposals (rank) or compare_policies (compare)
    audit: dict                   # compare: Auditor's rule check
    records: list                 # DecisionRecords logged / fetched / appealed by the Auditor
    recommend: dict               # recommend: matches and skills to learn
    rank_data: dict               # rank: tables shown before the Explainer writes
    pending: list                 # decision ids waiting for human approval
    answer: str
    data: dict
    decision_ids: list
    events: Annotated[list, operator.add]    # trace events (as dicts), appended by every node
    visited: Annotated[list, operator.add]   # node names in the order they ran


# ------------------------------------------------------------------ nodes --

def _ctx(config):
    return config["configurable"]["ctx"]


def _path(config) -> Path:
    return Path(config["configurable"]["decisions_path"])


def _node(name: str, fn):
    """Wrap a node so it also records its trace events and its name in the state."""
    def wrapped(state: State, config) -> dict:
        ctx = _ctx(config)
        before = len(ctx.trace.events) if ctx else 0
        update = fn(state, config) or {}
        new = [e.to_dict() for e in ctx.trace.events[before:]] if ctx else []
        return {**update, "events": new, "visited": [name]}
    wrapped.__name__ = name
    return wrapped


def planner_node(state: State, config) -> dict:
    return {"plan": planner.plan(state["request"], _ctx(config), state.get("profile"))}


def matcher_node(state: State, config) -> dict:
    plan, ctx = state["plan"], _ctx(config)
    if plan.intent == "rank":
        return {"match": matcher.rank(plan, ctx)}
    if plan.intent == "compare":
        allocations, sims = matcher.compare(plan, ctx)
        return {"allocations": allocations, "sims": sims}
    result = matcher.recommend(plan, ctx)                           # recommend
    ctx.on_partial(PartialResult("recommend", {"recommend": result}))   # show matches now
    return {"recommend": result}


def equity_node(state: State, config) -> dict:
    plan, ctx = state["plan"], _ctx(config)
    if plan.intent == "rank":
        return {"equity": equity.check_proposals(state["match"], ctx)}
    return {"equity": equity.compare_policies(state["allocations"], state["sims"], ctx)}


def auditor_node(state: State, config) -> dict:
    plan, ctx, path = state["plan"], _ctx(config), _path(config)
    if plan.intent == "rank":
        logged = auditor.verify_and_log(state["match"]["proposals"], path, ctx)
        data, ids = rank_tables(plan, state["match"], state["equity"], logged)
        ctx.on_partial(PartialResult("rank", dict(data), ids))      # tables can be shown now
        return {"records": logged, "rank_data": data}
    if plan.intent == "compare":
        return {"audit": auditor.check_allocations(state["allocations"], ctx)}
    if plan.intent == "explain":
        record = auditor.fetch(plan.decision_id, path, ctx)
    else:                                                           # appeal
        record = auditor.appeal(plan.decision_id, plan.appeal_reason, plan.student_id, path, ctx)
    return {"records": [record] if record else []}


def explainer_node(state: State, config) -> dict:
    plan, ctx = state["plan"], _ctx(config)
    pending = []
    if plan.intent == "rank":
        logged = state["records"]
        summaries, texts = explainer.explain_all(logged, ctx)      # ONE LLM call for all decisions
        answer, data, ids = format_rank(plan, state["match"], state["equity"], logged,
                                        state["rank_data"], summaries, texts)
        pending = list(ids)
    elif plan.intent == "compare":
        answer, data, ids = format_compare(plan, state["equity"], state["sims"], state["audit"], ctx)
    elif plan.intent == "recommend":
        result = state["recommend"]
        answer, data, ids = format_recommend(result, explainer.explain_matches(result, ctx))
    else:                                                           # explain / appeal
        record = state["records"][0]
        answer, data, ids = format_record(record, *explainer.explain(record, ctx))
    return {"answer": answer, "data": data, "decision_ids": ids, "pending": pending}


def respond_node(state: State, config) -> dict:
    """Short answers that need no further agents (help, or nothing to work on)."""
    plan, ctx = state["plan"], _ctx(config)
    if plan.intent == "help":
        answer, data, ids = planner.CAPABILITIES, {}, []
    elif plan.intent == "recommend":
        answer, data, ids = missing_skills(ctx)
    elif plan.intent == "rank":
        answer, data, ids = no_matches(state["match"]["found"])
    else:                                                           # explain / appeal: no record
        answer, data, ids = no_record(plan.intent)
    return {"answer": answer, "data": data, "decision_ids": ids, "pending": []}


def human_approval_node(state: State, config) -> dict:
    """
    Pause until a human approves one of the proposed decisions, then approve it with
    the normal tool. Loops back here until every decision from this run is approved.
    """
    path = _path(config)
    waiting = [i for i in state["pending"] if get_decision(i, path).status != "approved"]
    if not waiting:
        return {"pending": []}
    answer = interrupt({"pending": waiting})            # the run stops here until resumed
    chosen = answer.get("approve") if isinstance(answer, dict) else None
    if chosen in waiting:
        approve_decision(chosen, path)
        waiting.remove(chosen)
    return {"pending": waiting}


# ------------------------------------------------------------------ edges --
# Each router returns a ROUTE LABEL (so the exported diagram shows why an edge is
# taken); the dicts in build_graph() map each label to the next node.

def after_planner(state: State) -> str:
    plan = state["plan"]
    if plan.intent in ("rank", "compare") or (plan.intent == "recommend" and has_profile(plan)):
        return "rank / compare / recommend"
    if plan.intent in ("explain", "appeal"):
        return "explain / appeal"
    return "help, or no skills given"


def after_matcher(state: State) -> str:
    intent = state["plan"].intent
    if intent == "recommend":
        return "recommend"
    if intent == "rank" and not state["match"]["found"]["opportunities"]:
        return "no listings found"
    return "rank / compare"


def after_auditor(state: State) -> str:
    if state["plan"].intent in ("explain", "appeal") and not state["records"]:
        return "no such decision"
    return "checked"


def after_explainer(state: State) -> str:
    return "rank: decisions proposed" if state.get("pending") else "done"


def after_approval(state: State) -> str:
    return "more to approve" if state.get("pending") else "all approved"


def build_graph() -> StateGraph:
    g = StateGraph(State)
    for name, fn in [("planner", planner_node), ("matcher", matcher_node),
                     ("equity_analyst", equity_node), ("auditor", auditor_node),
                     ("explainer", explainer_node), ("respond", respond_node),
                     ("human_approval", human_approval_node)]:
        g.add_node(name, _node(name, fn))
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", after_planner, {
        "rank / compare / recommend": "matcher", "explain / appeal": "auditor",
        "help, or no skills given": "respond"})
    g.add_conditional_edges("matcher", after_matcher, {
        "rank / compare": "equity_analyst", "recommend": "explainer", "no listings found": "respond"})
    g.add_edge("equity_analyst", "auditor")
    g.add_conditional_edges("auditor", after_auditor, {"checked": "explainer", "no such decision": "respond"})
    g.add_conditional_edges("explainer", after_explainer, {
        "rank: decisions proposed": "human_approval", "done": END})
    g.add_conditional_edges("human_approval", after_approval, {
        "more to approve": "human_approval", "all approved": END})
    g.add_edge("respond", END)
    return g


# ---------------------------------------------------- checkpointer / cache --

# Our own dataclasses that may be stored in a checkpoint (LangGraph asks for an allow-list).
_ALLOWED = [("agents.planner", "Plan"), ("data_pipeline.schemas", "DecisionRecord"),
            ("data_pipeline.schemas", "Appeal"), ("data_pipeline.schemas", "Student"),
            ("data_pipeline.schemas", "Opportunity")]
_compiled: dict[str, Any] = {}
_lock = threading.Lock()


def checkpoint_path(decisions_path: Path) -> Path:
    """The SQLite checkpoint file lives next to the decisions file."""
    return Path(decisions_path).with_name("checkpoints.sqlite")


def _threads_path(decisions_path: Path) -> Path:
    """decision id -> id of the paused run that is waiting to approve it."""
    return Path(decisions_path).with_name("pending_approvals.json")


def get_app(decisions_path: Path = DEFAULT_PATH):
    """The compiled graph with a SQLite checkpointer (one per decisions file)."""
    db = checkpoint_path(decisions_path)
    with _lock:
        if str(db) not in _compiled:
            db.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(db, check_same_thread=False)
            saver = SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED))
            _compiled[str(db)] = build_graph().compile(checkpointer=saver)
        return _compiled[str(db)]


def _load_threads(decisions_path: Path) -> dict:
    p = _threads_path(decisions_path)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_threads(threads: dict, decisions_path: Path) -> None:
    p = _threads_path(decisions_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(threads, indent=2))


# -------------------------------------------------------------- running ----

def stream_run(request: str, provider: Optional[str] = None, decisions_path: Path = DEFAULT_PATH,
               client=_UNSET, profile: Optional[dict] = None,
               thread_id: Optional[str] = None) -> Iterator[Any]:
    """
    Run the graph and yield StepStarted / TraceEvent / PartialResult as they happen,
    then the AgentResult. A rank run stops at human_approval; its result is yielded
    at that point and the paused run waits in the checkpoint for resume_approval().
    """
    outside: list = []          # events emitted outside a node (start / done lines)

    def sink(item) -> None:
        try:
            writer = get_stream_writer()          # inside a node: into LangGraph's stream
        except RuntimeError:
            writer = None
        if writer is None:
            outside.append(item)
        else:
            writer(item)

    trace = Trace(sink, sink)
    ctx = start_run(request, provider, client, trace, "langgraph")
    ctx.on_partial = sink
    mode_start = ctx.mode
    yield from outside
    outside.clear()

    thread_id = thread_id or f"run-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id, "ctx": ctx,
                               "decisions_path": str(decisions_path)}}
    app = get_app(decisions_path)
    for mode, chunk in app.stream({"request": request, "profile": profile}, config,
                                  stream_mode=["custom", "updates"]):
        if mode == "custom" and isinstance(chunk, (StepStarted, TraceEvent, PartialResult)):
            yield chunk

    state = app.get_state(config).values
    if state.get("pending"):                       # paused for approval: remember where
        threads = _load_threads(decisions_path)
        threads.update({i: thread_id for i in state["pending"]})
        _save_threads(threads, decisions_path)
    result = finish_run(request, state["plan"], ctx, mode_start, state["answer"],
                        state["data"], state["decision_ids"])
    yield from outside
    yield result


def run_graph(request: str, **kwargs):
    """Run to the end (or the approval pause) and return the AgentResult."""
    *_, result = stream_run(request, **kwargs)
    return result


def resume_approval(decision_id: str, decisions_path: Path = DEFAULT_PATH) -> bool:
    """
    Approve `decision_id` by resuming the paused run that proposed it.
    Returns False if no paused run is waiting for it (e.g. a classic run, or an
    older decision): the caller then approves it directly with approve_decision().
    """
    threads = _load_threads(decisions_path)
    thread_id = threads.get(decision_id)
    if thread_id is None:
        return False
    app = get_app(decisions_path)
    config = {"configurable": {"thread_id": thread_id, "ctx": None,
                               "decisions_path": str(decisions_path)}}
    snapshot = app.get_state(config)
    if not snapshot.next or decision_id not in (snapshot.values.get("pending") or []):
        threads.pop(decision_id, None)             # stale entry: nothing paused for it
        _save_threads(threads, decisions_path)
        return False
    app.invoke(Command(resume={"approve": decision_id}), config)
    still = set(app.get_state(config).values.get("pending") or [])
    threads = {d: t for d, t in _load_threads(decisions_path).items()
               if t != thread_id or d in still}
    _save_threads(threads, decisions_path)
    return True


def is_paused(decision_id: str, decisions_path: Path = DEFAULT_PATH) -> bool:
    """True if a paused run is waiting for a human to approve this decision."""
    thread_id = _load_threads(decisions_path).get(decision_id)
    if thread_id is None:
        return False
    snapshot = get_app(decisions_path).get_state({"configurable": {"thread_id": thread_id}})
    return bool(snapshot.next) and decision_id in (snapshot.values.get("pending") or [])


def mermaid() -> str:
    """The graph as a Mermaid diagram (for docs/graph.md)."""
    return build_graph().compile().get_graph().draw_mermaid()


GRAPH_DOC = """# The FairMatch agent graph (LangGraph)

Generated from the compiled graph by `python -m agents.graph`; do not edit by hand.
Rendered image: [screenshots/langgraph.png](screenshots/langgraph.png).

```mermaid
{mermaid}
```

| Node | Calls (unchanged) |
|---|---|
| `planner` | `agents.planner.plan` (rules, or a checked LLM reading of the request) |
| `matcher` | `agents.matcher.rank` / `compare` / `recommend` |
| `equity_analyst` | `agents.equity.check_proposals` / `compare_policies` |
| `auditor` | `agents.auditor.verify_and_log` / `check_allocations` / `fetch` / `appeal` |
| `explainer` | `agents.explainer.explain_all` / `explain_matches` / `explain` + shared answer formatting |
| `respond` | short answers: the help text, "tell me your skills", "no such decision" |
| `human_approval` | `interrupt()`: pauses until a person presses **Approve**, then `tools.decisions.approve_decision` |

Solid arrows always run; dotted arrows are conditional routes, labelled with the intent
or outcome that takes them. The routes match the fixed steps in `agents/planner.py` (`STEPS`).
"""


if __name__ == "__main__":
    # Regenerate docs/graph.md and docs/screenshots/langgraph.png.
    # The PNG is drawn by the public mermaid.ink service (only the diagram text is sent).
    root = Path(__file__).resolve().parents[1] / "docs"
    (root / "graph.md").write_text(GRAPH_DOC.format(mermaid=mermaid().strip()))
    png = build_graph().compile().get_graph().draw_mermaid_png()
    (root / "screenshots" / "langgraph.png").write_bytes(png)
    print(f"wrote {root / 'graph.md'} and {root / 'screenshots' / 'langgraph.png'} ({len(png):,} bytes)")
