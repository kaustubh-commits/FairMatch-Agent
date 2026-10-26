"""
Shared pieces for every page: styling, the sidebar, the live agent trace,
and the decision card. Pages import from here so they all look the same.

Streamlit re-runs a page script top to bottom on every click; anything that
must survive a re-run lives in `st.session_state`.
"""

from __future__ import annotations

import html
import json
import os
from typing import Optional

import streamlit as st

from agents.orchestrator import AgentResult, stream
from agents.trace import AGENT_COLORS, AGENT_ICONS, PIPELINE, StepStarted, TraceEvent
from data_pipeline.build import PROCESSED_DIR
from llm.client import DEFAULT_GEMINI_MODEL, load_env
from tools.allocation import POLICIES
from tools.store import get_opportunity, load_opportunities, load_students

# ------------------------------------------------------------- colours ------
# Validated categorical palette (dark steps). Policies use the first three
# slots, which pass every colour-blindness check against our dark surface.
POLICY_COLORS = {"skill_first": "#3987e5", "equal_access": "#d95926", "hybrid": "#199e70"}
POLICY_LABELS = {"skill_first": "Skill-first", "equal_access": "Equal access", "hybrid": "Hybrid"}
DOMAIN_COLORS = {"ML": "#3987e5", "Data": "#d95926", "Web": "#199e70", "App": "#c98500",
                 "Cloud": "#d55181", "Design": "#9085e9", "Other": "#7d8597"}
STATUS_STYLES = {  # (colour, icon) - always shown with the word, never colour alone
    "proposed": ("#fab219", "●"), "approved": ("#0ca30c", "✓"),
    "appealed": ("#3987e5", "↺"), "revised": ("#9085e9", "✎"),
}
ACCENT = "#2dd4bf"

MODES = {"Gemini": "gemini", "Demo-safe": "demo-safe"}

CSS = """
<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.fm-title {display:flex; align-items:baseline; gap:.75rem; margin-bottom:.15rem;}
.fm-title h1 {font-size:1.9rem; font-weight:700; margin:0; padding:0;}
.fm-sdg {display:inline-block; font-size:.78rem; font-weight:600; color:#0b0f19; background:#2dd4bf;
         padding:.12rem .55rem; border-radius:999px;}
.fm-caption {color:#9aa3b5; font-size:.95rem; margin:.1rem 0 1.2rem 0;}
.fm-card {background:#131a2a; border:1px solid #232b3f; border-radius:.8rem; padding:1rem 1.1rem;}
.fm-muted {color:#9aa3b5;}
.fm-small {font-size:.82rem;}
.fm-chip {display:inline-block; font-size:.75rem; padding:.1rem .5rem; border-radius:999px;
          border:1px solid #2c3550; color:#c9cfdc; margin-right:.3rem; white-space:nowrap;}
.fm-badge {display:inline-flex; align-items:center; gap:.3rem; font-size:.75rem; font-weight:600;
           padding:.12rem .55rem; border-radius:999px; background:color-mix(in srgb, var(--c) 18%, transparent);
           color:var(--c); border:1px solid color-mix(in srgb, var(--c) 45%, transparent);}
.fm-note {border-left:3px solid var(--c, #2dd4bf); background:#131a2a; padding:.6rem .85rem;
          border-radius:.4rem; margin:.4rem 0; font-size:.92rem;}
/* --- agent pipeline --- */
.fm-pipe {display:flex; flex-wrap:wrap; gap:.35rem; margin:.2rem 0 .8rem 0;}
.fm-step {display:flex; align-items:center; gap:.35rem; font-size:.78rem; padding:.25rem .6rem;
          border-radius:999px; border:1px solid #2c3550; color:#6b7489; background:#0f1422;}
.fm-step.done {color:var(--c); border-color:color-mix(in srgb, var(--c) 50%, transparent);
               background:color-mix(in srgb, var(--c) 10%, transparent);}
.fm-step.active {color:#0b0f19; background:var(--c); border-color:var(--c); font-weight:600;
                 animation: fm-glow 1.2s ease-in-out infinite;}
@keyframes fm-glow {0%,100% {box-shadow:0 0 0 0 color-mix(in srgb, var(--c) 55%, transparent);}
                    50% {box-shadow:0 0 0 6px transparent;}}
/* --- trace events --- */
.fm-ev {border-left:3px solid var(--c); background:#131a2a; border-radius:.45rem;
        padding:.45rem .7rem; margin-bottom:.4rem;}
.fm-ev-head {display:flex; align-items:center; gap:.45rem; font-size:.82rem; flex-wrap:wrap;}
.fm-ev-agent {font-weight:700; color:var(--c);}
.fm-ev-action {color:#e6e8ef;}
.fm-ev-tool {font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.72rem; color:#9aa3b5;
             background:#0b0f19; border:1px solid #232b3f; padding:0 .35rem; border-radius:.3rem;}
.fm-ev-ms {margin-left:auto; color:#6b7489; font-size:.72rem;}
.fm-ev-body {color:#aeb6c6; font-size:.8rem; margin-top:.2rem; line-height:1.35; word-break:break-word;}
.fm-ev.warn {background:color-mix(in srgb, #fab219 10%, #131a2a);}
.fm-ev.live {border-left-style:dashed;}
.fm-dot {width:.55rem; height:.55rem; border-radius:50%; background:var(--c); display:inline-block;
         animation: fm-pulse 1s ease-in-out infinite;}
@keyframes fm-pulse {0%,100% {opacity:1; transform:scale(1);} 50% {opacity:.35; transform:scale(.7);}}
.fm-empty {color:#6b7489; font-size:.88rem; padding:1.5rem .5rem; text-align:center;}
/* --- skill bars --- */
.fm-skill {display:grid; grid-template-columns: 8.5rem 1fr 2.4rem; align-items:center; gap:.5rem;
           font-size:.78rem; margin:.15rem 0;}
.fm-bar {height:.42rem; background:#0b0f19; border-radius:999px; overflow:hidden; border:1px solid #232b3f;}
.fm-bar > span {display:block; height:100%; background:var(--c, #2dd4bf); border-radius:999px;}
.fm-kind {color:#6b7489; font-size:.7rem;}
/* --- sidebar --- */
section[data-testid="stSidebar"] .fm-brand {font-size:1.25rem; font-weight:800; letter-spacing:-.02em;}
section[data-testid="stSidebar"] .fm-brand span {color:#2dd4bf;}
.fm-sdg-row {display:flex; gap:.35rem; margin-top:.3rem;}
.fm-sdg-pill {font-size:.75rem; font-weight:700; padding:.15rem .5rem; border-radius:.35rem; color:white;}
</style>
"""


def esc(value) -> str:
    """HTML-escape any value (agent text can contain < or &)."""
    return html.escape(str(value))


# ------------------------------------------------------------ page shell ----

def setup_page() -> str:
    """Inject CSS and draw the sidebar. Returns the selected LLM provider."""
    st.html(CSS)
    return sidebar()


def page_header(title: str, sdg: str, caption: str) -> None:
    st.html(f'<div class="fm-title"><h1>{esc(title)}</h1><span class="fm-sdg">{esc(sdg)}</span></div>'
            f'<div class="fm-caption">{esc(caption)}</div>')


@st.cache_data
def data_badge() -> str:
    meta = json.loads((PROCESSED_DIR / "meta.json").read_text())
    n_students = len(load_students())
    if meta["opportunity_source"] == "kaggle":
        return f"{meta['n_opportunities']} real Internshala listings · {n_students} synthetic students"
    return f"{meta['n_opportunities']} synthetic listings · {n_students} synthetic students"


def _keep_a_mode() -> None:
    """Clicking the selected option would deselect it; restore the previous choice instead."""
    if st.session_state.mode is None:
        st.session_state.mode = st.session_state.last_mode
    st.session_state.last_mode = st.session_state.mode


def sidebar() -> str:
    load_env()
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    if "mode" not in st.session_state:
        st.session_state.mode = "Gemini" if has_key else "Demo-safe"
        st.session_state.last_mode = st.session_state.mode

    with st.sidebar:
        st.html('<div class="fm-brand">Fair<span>Match</span> Bangalore</div>'
                '<div class="fm-muted fm-small">Agentic, explainable internship allocation</div>')
        st.segmented_control("LLM mode", list(MODES), key="mode", on_change=_keep_a_mode,
                             help="Gemini reads requests and writes summaries. Demo-safe uses rules "
                                  "and templates only. Scores and decisions are identical in both.")
        mode = st.session_state.mode or st.session_state.last_mode
        if mode == "Gemini":
            model = os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
            status = (f"Model `{model}`. Falls back to demo-safe automatically if a call fails."
                      if has_key else "No GEMINI_API_KEY in .env, so runs will fall back to demo-safe.")
        else:
            status = "Rule-based planner and template explanations. No API calls."
        st.caption(status)

        st.divider()
        st.html(f'<div class="fm-muted fm-small" style="margin-bottom:.35rem">Data</div>'
                f'<span class="fm-badge" style="--c:{ACCENT}">● {esc(data_badge())}</span>')
        st.html('<div class="fm-muted fm-small" style="margin:.9rem 0 .1rem">UN Sustainable Development Goals</div>'
                '<div class="fm-sdg-row">'
                '<span class="fm-sdg-pill" style="background:#a21942">SDG 8</span>'
                '<span class="fm-sdg-pill" style="background:#dd1367">SDG 10</span>'
                '<span class="fm-sdg-pill" style="background:#00689d">SDG 16</span></div>'
                '<div class="fm-muted fm-small" style="margin-top:.35rem">Decent work · Reduced inequality · '
                'Accountable institutions</div>')
        st.caption("A human approves every decision.")
    return MODES[mode]


# ------------------------------------------------------------ live trace ----

def pipeline_html(done: set, active: Optional[str]) -> str:
    chips = []
    for agent in PIPELINE:
        state = "active" if agent == active else ("done" if agent in done else "")
        mark = " ✓" if state == "done" else ""
        chips.append(f'<span class="fm-step {state}" style="--c:{AGENT_COLORS[agent]}">'
                     f'{AGENT_ICONS[agent]} {esc(agent)}{mark}</span>')
    return '<div class="fm-pipe">' + "".join(chips) + "</div>"


def event_html(ev: TraceEvent) -> str:
    color = AGENT_COLORS.get(ev.agent, "#9ca3af")
    is_llm = bool(ev.tool and ev.tool.startswith("llm"))
    tool = "LLM · " + ev.tool[5:-1] if is_llm and "(" in (ev.tool or "") else ev.tool
    warn = ("fallback" in ev.action or "rejected" in ev.action
            or "FLAG" in ev.result_summary or ev.result_summary.startswith("FAILED"))
    ms = f'<span class="fm-ev-ms">{ev.duration_ms:,} ms</span>' if ev.duration_ms else ""
    body = f'<div class="fm-ev-body">{esc(ev.result_summary)}</div>' if ev.result_summary else ""
    tool_html = f'<span class="fm-ev-tool">{esc(tool)}</span>' if tool else ""
    return (f'<div class="fm-ev{" warn" if warn else ""}" style="--c:{color}">'
            f'<div class="fm-ev-head"><span>{AGENT_ICONS.get(ev.agent, "•")}</span>'
            f'<span class="fm-ev-agent">{esc(ev.agent)}</span>'
            f'<span class="fm-ev-action">{esc(ev.action)}</span>{tool_html}{ms}</div>{body}</div>')


def active_html(step: StepStarted) -> str:
    color = AGENT_COLORS.get(step.agent, "#9ca3af")
    verb = "is thinking" if (step.tool or "").startswith("llm") else "is working"
    tool = f' <span class="fm-ev-tool">{esc(step.tool)}</span>' if step.tool else ""
    return (f'<div class="fm-ev live" style="--c:{color}"><div class="fm-ev-head">'
            f'<span class="fm-dot"></span><span class="fm-ev-agent">{esc(step.agent)}</span>'
            f'<span class="fm-ev-action">{verb}: {esc(step.action)}</span>{tool}</div></div>')


def trace_html(events: list[TraceEvent], active: Optional[StepStarted] = None) -> str:
    done = {e.agent for e in events if e.agent in PIPELINE}
    parts = [pipeline_html(done, active.agent if active else None)]
    parts += [event_html(e) for e in events]
    if active:
        parts.append(active_html(active))
    if not events and not active:
        parts.append('<div class="fm-empty">The agents\' steps will stream here as they happen.</div>')
    return "".join(parts)


def run_live(request: str, provider: str, slot) -> AgentResult:
    """Run the agents and redraw the trace in `slot` after every step."""
    events: list[TraceEvent] = []
    for item in stream(request, provider=provider):
        if isinstance(item, StepStarted):
            slot.html(trace_html(events, active=item))
        elif isinstance(item, TraceEvent):
            events.append(item)
            slot.html(trace_html(events))
        else:
            return item
    raise RuntimeError("agent run ended without a result")


# --------------------------------------------------------- decision card ----

def status_badge(status: str) -> str:
    color, icon = STATUS_STYLES.get(status, ("#9ca3af", "•"))
    return f'<span class="fm-badge" style="--c:{color}">{icon} {esc(status.title())}</span>'


def policy_chip(policy: str) -> str:
    return (f'<span class="fm-chip"><span style="color:{POLICY_COLORS.get(policy, "#9ca3af")}">●</span> '
            f'{esc(POLICY_LABELS.get(policy, policy))} policy</span>')


def skill_bars(breakdown: dict, opp) -> str:
    """Per-skill bars from a record's {skill: points} breakdown (level = points / weight x 3)."""
    rows = []
    for skill, points in breakdown.items():
        if skill == "score":
            continue
        required = skill in opp.required_skills
        weight = 2 if required else 1
        level = round(points / weight * 3)
        rows.append(f'<div class="fm-skill"><span>{esc(skill)} <span class="fm-kind">'
                    f'{"required" if required else "preferred"}</span></span>'
                    f'<div class="fm-bar"><span style="width:{level / 3 * 100:.0f}%"></span></div>'
                    f'<span class="fm-muted">{level}/3</span></div>')
    return "".join(rows)


def candidate_block(sid: str, record: dict, opp, role: str) -> str:
    data = record["score_breakdown"].get(sid, {})
    score = data.get("score")
    past = next((s.past_opportunities for s in load_students() if s.id == sid), None)
    fit = f"fit {score:.2f}" if score is not None else ""
    return (f'<div style="margin-bottom:.6rem"><div style="display:flex;gap:.5rem;align-items:baseline">'
            f'<b>{esc(sid)}</b><span class="fm-muted fm-small">{esc(role)} · {fit} · '
            f'{past} past opportunities</span></div>{skill_bars(data, opp)}</div>')


def decision_header(record: dict) -> str:
    opp = get_opportunity(record["opportunity_id"])
    when = (record.get("timestamp") or "")[:16].replace("T", " ")
    return (f'<div style="display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;align-items:center">'
            f'<div><div style="font-size:1.05rem;font-weight:700">{esc(record["id"])} · {esc(opp.title)}</div>'
            f'<div class="fm-muted fm-small">{esc(opp.company)} · {esc(opp.area)} · {esc(opp.domain)} · '
            f'{opp.slots} slot(s) · {esc(when)} UTC</div></div>'
            f'<div style="display:flex;gap:.4rem">{policy_chip(record["policy"])}{status_badge(record["status"])}</div>'
            f'</div>')


def decision_body(record: dict) -> str:
    opp = get_opportunity(record["opportunity_id"])
    selected = "".join(candidate_block(sid, record, opp, "selected") for sid in record["selected"])
    runner = (candidate_block(record["runner_up"], record, opp, "runner-up")
              if record.get("runner_up") and record["runner_up"] not in record["selected"] else
              '<div class="fm-muted fm-small">No runner-up.</div>')
    return (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1.2rem;'
            f'margin-top:.8rem"><div><div class="fm-muted fm-small" style="margin-bottom:.3rem">SELECTED</div>'
            f'{selected or "<span class=fm-muted>Nobody</span>"}</div>'
            f'<div><div class="fm-muted fm-small" style="margin-bottom:.3rem">RUNNER-UP</div>{runner}</div></div>')


def appeal_html(record: dict) -> str:
    appeal = record.get("appeal")
    if not appeal:
        return ""
    color = "#9085e9" if appeal["outcome"] == "revised" else "#3987e5"
    return (f'<div class="fm-note" style="--c:{color}"><b>Appeal by {esc(appeal["student_id"])}</b> '
            f'<span class="fm-muted">"{esc(appeal["text"])}"</span><br>'
            f'<b>Auditor: {esc(appeal["outcome"])}.</b> {esc(appeal["explanation"])}</div>')
