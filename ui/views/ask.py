"""
Ask page (coordinators): chat with the agents on the left, what they did on the right.

Flow of one request:
  1. the request comes from the chat box or an example button
  2. run_live() iterates agents.orchestrator.stream(); every step redraws the trace
  3. a PartialResult arrives once the tables are ready -> they are drawn at once,
     with "Writing the explanation…" underneath
  4. the AgentResult arrives -> the (already verified) explanation is revealed
"""

import streamlit as st

from agents.orchestrator import AgentResult
from tools.decisions import get_decision
from tools.store import get_opportunity
from ui.common import (POLICY_LABELS, appeal_html, decision_body, decision_header, dev_toggle, esc,
                       note_html, page_header, plural, reveal, run_live, show_recommendations,
                       trace_html, writing)

EXAMPLES = [  # the four demo-script requests (PLAN.md section 2)
    ("🎯 Rank ML candidates", "Find the best 3 candidates for the ML internships in Koramangala and explain why."),
    ("⚖️ Compare policies", "Compare the allocation policies and tell me which is fairest."),
    ("💬 Explain a decision", "Explain the most recent decision."),
    ("🛡️ Appeal a decision", "Appeal the most recent decision on behalf of the runner-up: "
                             "they have had no internships yet and have more hours available."),
]

page_header("Ask the agents",
            "Rank candidates, compare policies, or explain and appeal a decision. "
            "Every step the agents take is listed on the right.")

if "chat" not in st.session_state:
    st.session_state.chat = []          # list of {"role": "user"/"assistant", "text" or "result"}


# ---------------------------------------------------------- answer views ----

def show_rank(data: dict, decision_ids: list[str], summaries: dict | None = None) -> dict:
    """Candidate tables per decision. Returns {decision_id: explanation placeholder}."""
    if data.get("found_note"):
        st.html(note_html(data["found_note"], "📍", "#60a5fa"))
    slots = {}
    for did in decision_ids:
        record = get_decision(did)
        opp = get_opportunity(record.opportunity_id)
        with st.container(border=True):
            st.html(f'<div style="font-weight:700">{esc(opp.title)}</div>'
                    f'<div class="fm-muted fm-small">{esc(opp.company)} · {esc(opp.area)} · '
                    f'{plural(opp.slots, "slot")} · decision {esc(did)}</div>')
            rows = [{"Student": m["student_label"], "Match": m["score"], "Skill fit": m["skill_fit"],
                     "Commute": m["commute_km"], "Past opportunities": m["past_opportunities"]}
                    for m in data["rankings"].get(opp.id, [])]
            st.dataframe(rows, hide_index=True, width="stretch", column_config={
                "Student": st.column_config.TextColumn(width="medium"),
                "Match": st.column_config.ProgressColumn("Match", min_value=0, max_value=1, format="%.2f",
                                                         width="small"),
                "Skill fit": st.column_config.NumberColumn(format="%.2f", width="small"),
                "Commute": st.column_config.NumberColumn(format="%.1f km", width="small"),
                "Past opportunities": st.column_config.NumberColumn(width="small"),
            })
            slots[did] = st.empty()
            if summaries and did in summaries:
                slots[did].html(note_html(summaries[did]))
            else:
                writing(slots[did])
    equity = data.get("equity", {})
    if equity.get("message"):
        st.html(note_html(equity["message"], "⚖️", "#fbbf24" if equity.get("flagged") else "#34d399"))
    if decision_ids:
        st.page_link("views/decision_log.py", label="Approve or appeal in the Decision Log",
                     icon=":material/fact_check:")
    return slots


def show_compare(result: AgentResult) -> None:
    metrics = result.data["metrics"]
    rows = [{"Policy": POLICY_LABELS[p], "Skill fit": m["avg_fit"], "Gini": m["gini"],
             "Students reached": m["unique_recipients"],
             "Repeat selection": f"{m['repeat_selection_rate']:.0%}",
             "Avg commute": f"{m['avg_commute_km']} km", "Outskirts gap": f"{m['outskirts_gap_pts']} pts"}
            for p, m in metrics.items()]
    st.dataframe(rows, hide_index=True, width="stretch")
    st.html(note_html(result.data["takeaway"], "⚖️", "#fbbf24"))
    flagged = bool(result.data.get("outskirts_flags"))
    st.html(note_html(result.data["outskirts"], "📍", "#fbbf24" if flagged else "#34d399"))
    st.page_link("views/policy_lab.py", label="See the charts in the Policy Lab", icon=":material/balance:")


def show_record(result: AgentResult) -> None:
    record = result.data["record"]
    with st.container(border=True):
        st.html(decision_header(record) + decision_body(record) + appeal_html(record))
    if result.data.get("summary"):
        st.html(note_html(result.data["summary"]))


def show_result(result: AgentResult) -> None:
    """A finished answer, drawn from the stored result (used for chat history)."""
    if result.intent == "rank" and result.decision_ids:
        show_rank(result.data, result.decision_ids, result.data.get("summaries"))
    elif result.intent == "compare":
        show_compare(result)
    elif result.intent == "recommend" and result.data.get("recommend"):
        show_recommendations(result.data["recommend"], result.data["summary"])
    elif result.intent in ("explain", "appeal") and "record" in result.data:
        show_record(result)
    else:
        st.markdown(result.answer)
    if st.session_state.get("dev"):
        fell_back = "" if result.mode_end == result.mode_start else f" → {result.mode_end}"
        st.caption(f"intent: {result.intent} · mode: {result.mode_start}{fell_back}")


# ---------------------------------------------------------------- layout ----

left, right = st.columns([1.3, 1], gap="large")

with left:
    chat_box = st.container(height=520, border=False)
    clicked = None
    for row in (EXAMPLES[:2], EXAMPLES[2:]):
        for col, (text_label, text) in zip(st.columns(2), row):
            if col.button(text_label, width="stretch"):
                clicked = text
    typed = st.chat_input("Ask about candidates, policies or a decision…")

with right:
    head, toggle = st.columns([1.4, 1])
    head.html('<div style="font-weight:700;padding-top:6px">What the agents did</div>')
    with toggle:
        dev = dev_toggle("ask")
    with st.container(height=640, border=True):
        trace_slot = st.empty()
    last = next((m["result"] for m in reversed(st.session_state.chat) if "result" in m), None)
    trace_slot.html(trace_html(last.events if last else [], dev=dev))

with chat_box:
    if not st.session_state.chat:
        st.html('<div class="fm-card fm-muted fm-small">Try one of the requests below, or type your own. '
                'Five agents work through it: they read the request, score students with fixed rules, '
                'check fairness, enforce the rules and explain the result.</div>')
    for message in st.session_state.chat:
        with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "⚖️"):
            if "result" in message:
                show_result(message["result"])
            else:
                st.markdown(message["text"])

request = typed or clicked
if request:
    st.session_state.chat.append({"role": "user", "text": request})
    with chat_box:
        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(request)
        answer = st.chat_message("assistant", avatar="⚖️").container()
    with answer:
        status = st.empty()
        writing(status, "The agents are working")
    pending = {}     # explanation placeholders created by the partial result

    def on_partial(partial) -> None:
        """Draw the tables as soon as they exist, before the explanation is written."""
        status.empty()
        with answer:
            if partial.intent == "rank":
                pending.update(show_rank(partial.data, partial.decision_ids))
            elif partial.intent == "recommend":
                pending["recommend"] = show_recommendations(partial.data["recommend"])

    result = run_live(request, st.session_state.provider, trace_slot, on_partial=on_partial)
    status.empty()
    if result.intent == "rank" and pending:
        for did, slot in pending.items():
            reveal(slot, result.data["summaries"][did])
    elif result.intent == "recommend" and "recommend" in pending:
        reveal(pending["recommend"], result.data["summary"])
    else:
        with answer:
            show_result(result)
    st.session_state.chat.append({"role": "assistant", "result": result})
