"""
Ask page: chat with the agents on the left, live agent trace on the right.

Flow of one request:
  1. the request comes from the chat box or an example button
  2. run_live() iterates agents.orchestrator.stream(); every step redraws the trace
  3. the AgentResult is stored in the chat history and the page re-runs
"""

import streamlit as st

from agents.orchestrator import AgentResult
from tools.decisions import get_decision
from tools.store import get_opportunity
from ui.common import (POLICY_LABELS, appeal_html, decision_body, decision_header, esc,
                       page_header, run_live, show_recommendations, trace_html)

EXAMPLES = [  # the four demo-script requests (PLAN.md section 2)
    ("🎯 Rank ML candidates", "Find the best 3 candidates for the ML internships in Koramangala and explain why."),
    ("⚖️ Compare policies", "Compare the allocation policies and tell me which is fairest."),
    ("💬 Explain a decision", "Explain the most recent decision."),
    ("🛡️ Appeal a decision", "Appeal the most recent decision on behalf of the runner-up: "
                             "they have had no internships yet and have more hours available."),
]

page_header("Ask FairMatch",
            "Ask the agents to rank candidates, compare policies, explain or appeal a decision, "
            "and watch every step they take.")

if "chat" not in st.session_state:
    st.session_state.chat = []          # list of {"role": "user"/"assistant", "text" or "result"}


# ---------------------------------------------------------- answer views ----

def show_rank(result: AgentResult) -> None:
    data = result.data
    if data.get("found_note"):
        st.html(f'<div class="fm-note" style="--c:#60a5fa">📍 {esc(data["found_note"])}</div>')
    summaries = data.get("summaries", {})
    for did in result.decision_ids:
        record = get_decision(did)
        opp = get_opportunity(record.opportunity_id)
        with st.container(border=True):
            st.html(f'<div style="font-weight:700">{esc(opp.title)}</div>'
                    f'<div class="fm-muted fm-small">{esc(opp.company)} · {esc(opp.area)} · '
                    f'{opp.slots} slot(s) · decision {esc(did)} ({esc(record.status)})</div>')
            rows = [{"#": i, "Student": m["student_id"], "Fit": m["score"],
                     "Past opps": m["past_opportunities"],
                     "Skills": " · ".join(f"{s} {b['level']}/3" for s, b in m["breakdown"].items())}
                    for i, m in enumerate(data["rankings"].get(opp.id, []), start=1)]
            st.dataframe(rows, hide_index=True, width="stretch", column_config={
                "#": st.column_config.NumberColumn(width=40),
                "Student": st.column_config.TextColumn(width=80),
                "Skills": st.column_config.TextColumn(width="large"),
                "Fit": st.column_config.ProgressColumn("Skill fit", min_value=0, max_value=1, format="%.2f",
                                                       width=130),
                "Past opps": st.column_config.NumberColumn("Past", width=60),
            })
            if did in summaries:
                st.html(f'<div class="fm-small">💬 {esc(summaries[did])}</div>')
    equity = data.get("equity", {})
    if equity.get("message"):
        color = "#fbbf24" if equity.get("flagged") else "#34d399"
        st.html(f'<div class="fm-note" style="--c:{color}">⚖️ {esc(equity["message"])}</div>')
    if result.decision_ids:
        st.page_link("views/decision_log.py", label="Review, approve or appeal in the Decision Log",
                     icon=":material/fact_check:")


def show_compare(result: AgentResult) -> None:
    metrics = result.data["metrics"]
    rows = [{"Policy": POLICY_LABELS[p], "Avg skill fit": m["avg_fit"], "Gini": m["gini"],
             "Unique recipients": m["unique_recipients"],
             "Repeat selection": f"{m['repeat_selection_rate']:.1%}"} for p, m in metrics.items()]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.html(f'<div class="fm-note" style="--c:#fbbf24">⚖️ {esc(result.data["takeaway"])}</div>')
    st.html(f'<div class="fm-note" style="--c:#fbbf24">🔁 {esc(result.data["rounds_takeaway"])}</div>')
    st.page_link("views/policy_lab.py", label="See the charts in the Policy Lab", icon=":material/balance:")


def show_record(result: AgentResult) -> None:
    record = result.data["record"]
    with st.container(border=True):
        st.html(decision_header(record) + decision_body(record) + appeal_html(record))
    if result.data.get("summary"):
        st.html(f'<div class="fm-small">💬 {esc(result.data["summary"])}</div>')


def show_result(result: AgentResult) -> None:
    if result.intent == "rank" and result.decision_ids:
        show_rank(result)
    elif result.intent == "compare":
        show_compare(result)
    elif result.intent == "recommend" and result.data.get("recommend"):
        show_recommendations(result.data["recommend"], result.data["summary"])
        st.page_link("views/students.py", label="Try different skills on the For Students page",
                     icon=":material/school:")
    elif result.intent in ("explain", "appeal") and "record" in result.data:
        show_record(result)
    else:
        st.markdown(result.answer)
    def pretty(mode: str) -> str:
        """'gemini:gemini-3.5-flash (fallbacks: ...)' -> 'Gemini (gemini-3.5-flash)'."""
        if mode.startswith("gemini:"):
            return f"Gemini ({mode.split(':', 1)[1].split(' ')[0]})"
        return mode
    mode = pretty(result.mode_start) if result.mode_end == result.mode_start else \
        f"{pretty(result.mode_start)} → fell back to {result.mode_end}"
    st.caption(f"Intent: {result.intent} · Mode: {mode}")


# ---------------------------------------------------------------- layout ----

left, right = st.columns([1.25, 1], gap="large")

with left:
    chat_box = st.container(height=480, border=False)
    st.html('<div class="fm-muted fm-small">Try a demo request</div>')
    clicked = None
    for row in (EXAMPLES[:2], EXAMPLES[2:]):
        for col, (label, text) in zip(st.columns(2), row):
            if col.button(label, width="stretch"):
                clicked = text
    st.page_link("views/students.py", label="Are you a student? See which internships fit your skills",
                 icon=":material/school:")
    typed = st.chat_input("Ask about candidates, policies or a decision…")

with right:
    st.html('<div style="font-weight:700;margin-bottom:.4rem">Live agent trace</div>')
    trace_box = st.container(height=700, border=True)
    with trace_box:
        trace_slot = st.empty()
    last = next((m["result"] for m in reversed(st.session_state.chat) if "result" in m), None)
    trace_slot.html(trace_html(last.events if last else []))

with chat_box:
    if not st.session_state.chat:
        st.html('<div class="fm-card" style="margin-top:.5rem"><b>How it works</b>'
                '<div class="fm-muted fm-small" style="margin-top:.35rem">A <b>Planner</b> reads your request, '
                'the <b>Matcher</b> scores students with deterministic tools, the <b>Equity Analyst</b> checks '
                'whether opportunities concentrate on the same students, the <b>Auditor</b> enforces the rules '
                'and logs a decision for human approval, and the <b>Explainer</b> writes only from that record. '
                'The LLM never produces a score or a reason.<br><br>Students can also ask, e.g. '
                '<i>"I know Python and SQL, what can I do?"</i></div></div>')
    for message in st.session_state.chat:
        with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "⚖️"):
            if "result" in message:
                show_result(message["result"])
            else:
                st.markdown(message["text"])

request = typed or clicked
if request:
    st.session_state.chat.append({"role": "user", "text": request})
    with chat_box:
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(request)
        with st.chat_message("assistant", avatar="⚖️"):
            st.html('<div class="fm-muted"><span class="fm-dot" style="--c:#2dd4bf"></span> '
                    'Agents working… follow along in the trace →</div>')
    result = run_live(request, st.session_state.provider, trace_slot)
    st.session_state.chat.append({"role": "assistant", "result": result})
    st.rerun()
