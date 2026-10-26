"""
Decision Log: every decision the agents proposed, with Approve and Appeal.

Approve is a plain human action (tools.decisions.approve_decision).
Appeal goes through the agents, so the Auditor re-evaluates the decision
deterministically and the Explainer reports the outcome, with a live trace.
"""

import streamlit as st

from tools.decisions import approve_decision, list_decisions
from tools.store import load_students
from ui.common import (STATUS_STYLES, appeal_html, decision_body, decision_header, esc, label,
                       note_html, page_header, run_live)

DEMO_QUERY = "Find the best 3 candidates for the ML internships in Koramangala and explain why."

page_header("Decision Log",
            "Every proposed decision with its reasons. Approve it, or appeal it and let the Auditor re-check.")

provider = st.session_state.provider


@st.dialog("Appeal a decision", width="large")
def appeal_dialog(record) -> None:
    st.html(f'<div class="fm-muted fm-small">Decision {esc(record.id)} · currently selected: '
            f'{esc(", ".join(label(s) for s in record.selected) or "nobody")}</div>')
    others = sorted((s for s in load_students() if s.id not in record.selected), key=lambda s: s.id)
    ids = [s.id for s in others]
    default = ids.index(record.runner_up) if record.runner_up in ids else 0
    with st.form("appeal_form", border=False):
        student = st.selectbox("Appealing student", ids, index=default, format_func=label,
                               help="Defaults to the runner-up.")
        reason = st.text_area("Reason", placeholder="e.g. I have had no internships yet and 15 hours a week free.")
        submitted = st.form_submit_button("Submit appeal", type="primary", width="stretch")
    if submitted:
        if not reason.strip():
            st.warning("Please give a reason.")
            return
        st.html('<div class="fm-label" style="margin-top:var(--sp-3)">What the agents did</div>')
        slot = st.empty()
        result = run_live(f"Appeal {record.id} for {student}: {reason.strip()}", provider, slot)
        updated = result.data.get("record")
        if updated and updated.get("appeal"):
            outcome = updated["appeal"]["outcome"]
            (st.success if outcome == "revised" else st.info)(
                f"**Auditor: {outcome}.** {updated['appeal']['explanation']}")
            if result.data.get("summary"):
                st.html(note_html(result.data["summary"]))
        else:
            st.error(result.answer)
        if st.button("Close and refresh the log", width="stretch"):
            st.rerun()


records = list(reversed(list_decisions()))   # newest first

if not records:
    st.html('<div class="fm-card"><b>No decisions yet.</b><div class="fm-muted" style="margin-top:var(--sp-1)">'
            'Decisions appear here when the agents rank candidates on the Ask page, or you can '
            'generate the demo decisions.</div></div>')
    if st.button("Generate demo decisions", type="primary", icon=":material/play_arrow:"):
        with st.status("Agents ranking ML candidates…", expanded=True):
            run_live(DEMO_QUERY, provider, st.empty())
        st.rerun()
    st.stop()

# ------------------------------------------------- filter (with counts) ----
counts = {s: sum(1 for r in records if r.status == s) for s in STATUS_STYLES}
show = st.pills("Show", list(STATUS_STYLES), selection_mode="multi", default=list(STATUS_STYLES),
                format_func=lambda s: f"{STATUS_STYLES[s][1]} {s.title()} · {counts[s]}",
                label_visibility="collapsed")

# --------------------------------------------------------------- cards ----
for record in records:
    if record.status not in (show or []):
        continue
    data = record.to_dict()
    with st.container(border=True):
        st.html(decision_header(data) + decision_body(data) + appeal_html(data))
        with st.expander("Reasons, from the decision record"):
            st.html("".join(f'<div class="fm-small" style="margin:var(--sp-1) 0">• {esc(r)}</div>'
                            for r in record.reasons)
                    + '<div class="fm-muted fm-small" style="margin-top:var(--sp-2)">Rules: '
                    + esc(", ".join(f"{k}={v}" for k, v in record.constraints.items() if k != "skipped"))
                    + "</div>")
        a, b, _ = st.columns([1, 1, 3])
        if a.button("Approve", key=f"approve-{record.id}", icon=":material/check:",
                    type="primary", disabled=record.status == "approved", width="stretch"):
            approve_decision(record.id)
            st.toast(f"{record.id} approved", icon="✅")
            st.rerun()
        if b.button("Appeal", key=f"appeal-{record.id}", icon=":material/gavel:",
                    disabled=record.status == "approved", width="stretch"):
            appeal_dialog(record)
