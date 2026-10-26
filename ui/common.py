"""
Shared pieces for every page: styling, the sidebar, the live agent trace,
decision cards and recommendation cards. Pages import from here so they all
look the same.

Streamlit re-runs a page script top to bottom on every click; anything that
must survive a re-run lives in `st.session_state`.
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from agents.orchestrator import AgentResult, stream
from agents.trace import AGENT_COLORS, AGENT_ICONS, PIPELINE, PartialResult, StepStarted, TraceEvent
from data_pipeline.build import PROCESSED_DIR
from llm.client import DEFAULT_GEMINI_FALLBACK, DEFAULT_GEMINI_MODEL, load_env
from tools.decisions import BREAKDOWN_FIELDS
from tools.store import get_opportunity, get_student, load_opportunities, load_students

ASSETS = Path(__file__).resolve().parent / "assets"

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
WARN = "#fab219"

MODES = {"Gemini": "gemini", "Demo-safe": "demo-safe"}

# One type scale and one spacing scale, used everywhere (see docs/ui-critique.md).
CSS = """
<style>
:root {
  --fs-xs: .78rem; --fs-sm: .875rem; --fs-md: 1rem; --fs-lg: 1.25rem; --fs-xl: 1.75rem;
  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px; --sp-5: 24px;
  --surface: #131a2a; --surface-2: #0f1422; --border: #232b3f; --border-strong: #2c3550;
  --text: #e6e8ef; --muted: #9aa3b5; --faint: #6b7489; --accent: #2dd4bf;
  --ease: cubic-bezier(.2, .8, .2, 1);
}
.block-container {padding-top: var(--sp-5); padding-bottom: 48px; max-width: 1320px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.fm-title h1 {font-size: var(--fs-xl); font-weight: 700; margin: 0; padding: 0;}
.fm-caption {color: var(--muted); font-size: var(--fs-md); margin: var(--sp-1) 0 var(--sp-5) 0;}
.fm-h2 {font-size: var(--fs-lg); font-weight: 700; margin: var(--sp-5) 0 var(--sp-1) 0;}
.fm-label {color: var(--faint); font-size: var(--fs-xs); letter-spacing: .04em; text-transform: uppercase;
           margin-bottom: var(--sp-2);}
.fm-card {background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: var(--sp-4);}
.fm-muted {color: var(--muted);}
.fm-small {font-size: var(--fs-sm);}
.fm-chip {display: inline-flex; align-items: center; gap: var(--sp-1); font-size: var(--fs-xs);
          padding: 2px var(--sp-2); border-radius: 999px; border: 1px solid var(--border-strong); color: #c9cfdc;}
.fm-badge {display: inline-flex; align-items: center; gap: var(--sp-1); font-size: var(--fs-xs); font-weight: 600;
           padding: 2px var(--sp-2); border-radius: 999px; background: color-mix(in srgb, var(--c) 16%, transparent);
           color: var(--c); border: 1px solid color-mix(in srgb, var(--c) 40%, transparent);}
.fm-note {border-left: 3px solid var(--c, var(--accent)); background: var(--surface);
          padding: var(--sp-3) var(--sp-4); border-radius: 8px; margin: var(--sp-2) 0;
          font-size: var(--fs-sm); line-height: 1.5;}
.fm-cursor {display: inline-block; width: .5em; color: var(--accent);}

/* --- pipeline + trace --- */
.fm-pipe {display: flex; flex-wrap: wrap; gap: var(--sp-1); margin-bottom: var(--sp-3);}
.fm-step {display: inline-flex; align-items: center; gap: var(--sp-1); font-size: var(--fs-xs);
          padding: 2px var(--sp-2); border-radius: 999px; border: 1px solid var(--border); color: var(--faint);
          transition: color 160ms var(--ease), border-color 160ms var(--ease), background-color 160ms var(--ease);}
.fm-step.done {color: var(--c); border-color: color-mix(in srgb, var(--c) 45%, transparent);}
.fm-step.active {color: var(--text); border-color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent);}
.fm-line {display: grid; grid-template-columns: 22px 1fr; gap: var(--sp-2); padding: var(--sp-2) 0;
          border-bottom: 1px solid var(--border); font-size: var(--fs-sm); line-height: 1.45;}
.fm-line:last-child {border-bottom: 0;}
.fm-line b {color: var(--c);}
.fm-line.warn {background: color-mix(in srgb, #fab219 8%, transparent); border-radius: 6px; padding-left: var(--sp-1);}
.fm-ev {border-left: 3px solid var(--c); background: var(--surface); border-radius: 6px;
        padding: var(--sp-2) var(--sp-3); margin-bottom: var(--sp-2);}
.fm-ev-head {display: flex; align-items: center; gap: var(--sp-2); font-size: var(--fs-xs); flex-wrap: wrap;}
.fm-ev-agent {font-weight: 700; color: var(--c);}
.fm-ev-tool {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: var(--fs-xs); color: var(--muted);
             background: #0b0f19; border: 1px solid var(--border); padding: 0 var(--sp-1); border-radius: 4px;}
.fm-ev-ms {margin-left: auto; color: var(--faint);}
.fm-ev-body {color: #aeb6c6; font-size: var(--fs-xs); margin-top: var(--sp-1); line-height: 1.4; word-break: break-word;}
.fm-ev.warn {background: color-mix(in srgb, #fab219 8%, var(--surface));}
.fm-dot {width: 8px; height: 8px; border-radius: 50%; background: var(--c); display: inline-block;
         margin-top: 6px; animation: fm-pulse 1.1s ease-in-out infinite;}
.fm-new {animation: fm-in 160ms var(--ease) both;}
@keyframes fm-in {from {opacity: 0; transform: translateY(4px);} to {opacity: 1; transform: none;}}
@keyframes fm-pulse {0%, 100% {opacity: 1;} 50% {opacity: .35;}}
.fm-empty {color: var(--faint); font-size: var(--fs-sm); padding: var(--sp-5) var(--sp-2); text-align: center;}
.fm-details summary {cursor: pointer; color: var(--faint); font-size: var(--fs-xs); margin-top: var(--sp-1);}
.fm-details pre {white-space: pre-wrap; word-break: break-word; font-size: var(--fs-xs); color: var(--muted);
                 background: #0b0f19; border: 1px solid var(--border); border-radius: 6px;
                 padding: var(--sp-2); margin: var(--sp-1) 0 0 0; max-height: 12rem; overflow: auto;}

/* --- skills, matches, learning --- */
.fm-skill {display: grid; grid-template-columns: 9rem 1fr 2.4rem; align-items: center; gap: var(--sp-2);
           font-size: var(--fs-xs); margin: 3px 0;}
.fm-bar {height: 6px; background: #0b0f19; border-radius: 999px; overflow: hidden; border: 1px solid var(--border);}
.fm-bar > span {display: block; height: 100%; background: var(--c, var(--accent)); border-radius: 999px;}
.fm-kind {color: var(--faint);}
.fm-match {display: grid; grid-template-columns: 1fr 6.5rem; gap: var(--sp-3); align-items: start;}
.fm-score {font-size: var(--fs-lg); font-weight: 700; text-align: right; line-height: 1.1;}
.fm-learn {display: grid; grid-template-columns: 8.5rem 1fr 3rem; gap: var(--sp-3); align-items: center;
           font-size: var(--fs-sm); margin: var(--sp-2) 0;}
.fm-person {display: flex; gap: var(--sp-2); align-items: baseline; flex-wrap: wrap; margin-bottom: var(--sp-1);}

/* --- home --- */
.fm-hero {font-size: 2.4rem; font-weight: 800; letter-spacing: -.02em; margin: var(--sp-5) 0 var(--sp-2) 0;}
.fm-hero span {color: var(--accent);}
.fm-lede {font-size: var(--fs-lg); color: var(--muted); max-width: 46rem; line-height: 1.45;}
.fm-path-title {font-size: var(--fs-lg); font-weight: 700; margin-bottom: var(--sp-1);}

/* --- light motion on controls: quick feedback, nothing decorative --- */
.stButton button, .stFormSubmitButton button, [data-testid="stPageLink"] a {
  transition: background-color 120ms var(--ease), border-color 120ms var(--ease), transform 80ms var(--ease);
}
.stButton button:active, .stFormSubmitButton button:active {transform: scale(.98);}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.fm-path) {transition: border-color 120ms var(--ease);}
div[data-testid="stVerticalBlockBorderWrapper"]:has(.fm-path):hover {border-color: var(--accent);}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {animation: none !important; transition: none !important;}
}
</style>
"""


def esc(value) -> str:
    """HTML-escape any value (agent text can contain < or &)."""
    return html.escape(str(value))


def label(sid: Optional[str]) -> str:
    """'Ananya R. (S025)' for a student id (or the id itself if unknown)."""
    if not sid:
        return ""
    try:
        return get_student(sid).label
    except KeyError:
        return sid


def plural(n: int, word: str, many: Optional[str] = None) -> str:
    return f"{n} {word if n == 1 else (many or word + 's')}"


def past_text(n: Optional[int]) -> str:
    return "" if n is None else plural(n, "past opportunity", "past opportunities")


# ------------------------------------------------------------ page shell ----

def setup_page() -> str:
    """Inject CSS, the logo and the sidebar. Returns the selected LLM provider."""
    st.html(CSS)
    st.logo(str(ASSETS / "logo.svg"), size="large")
    return sidebar()


def page_header(title: str, caption: str) -> None:
    st.html(f'<div class="fm-title"><h1>{esc(title)}</h1></div><div class="fm-caption">{esc(caption)}</div>')


@st.cache_data
def data_line() -> str:
    meta = json.loads((PROCESSED_DIR / "meta.json").read_text())
    n_students = len(load_students())
    source = "real Internshala listings" if meta["opportunity_source"] == "kaggle" else "synthetic listings"
    return f"{meta['n_opportunities']} {source} · {n_students} synthetic students"


def _save_mode() -> None:
    """Copy the widget's choice into `llm_mode`. Clicking the selected option (None) keeps the old one."""
    if st.session_state.mode_widget:
        st.session_state.llm_mode = st.session_state.mode_widget


def sidebar() -> str:
    """Quiet sidebar: navigation comes first; data source and settings sit underneath."""
    load_env()
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    # The choice lives in `llm_mode`, a plain session key. Streamlit can drop a widget's own
    # state when switching pages, so the widget is re-filled from `llm_mode` on every run.
    st.session_state.setdefault("llm_mode", "Gemini" if has_key else "Demo-safe")
    st.session_state.mode_widget = st.session_state.llm_mode
    mode = st.session_state.llm_mode

    with st.sidebar:
        st.caption(data_line() + ". A human approves every decision.")
        with st.expander("Settings"):
            st.segmented_control("Language model", list(MODES), key="mode_widget", on_change=_save_mode,
                                 help="Gemini reads requests and writes summaries. Demo-safe uses rules "
                                      "and templates only. Scores and decisions are identical in both.")
            if mode == "Gemini":
                model = os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
                backup = os.environ.get("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK)
                st.caption(f"`{model}`, then `{backup}`, then rule-based." if has_key
                           else "No GEMINI_API_KEY in .env, so the rule-based mode is used.")
            else:
                st.caption("Rules and templates only. No API calls.")
    return MODES[mode]


def dev_toggle(key: str) -> bool:
    """The 'Developer details' switch. Its value is shared by every page."""
    value = st.toggle("Developer details", value=st.session_state.get("dev", False), key=f"dev-{key}",
                      help="Show tool names, timings, raw LLM output and error details.")
    st.session_state.dev = value
    return value


# ------------------------------------------------------------ live trace ----

def pipeline_html(done: set, active: Optional[str]) -> str:
    chips = []
    for agent in PIPELINE:
        state = "active" if agent == active else ("done" if agent in done else "")
        chips.append(f'<span class="fm-step {state}" style="--c:{AGENT_COLORS[agent]}">'
                     f'{AGENT_ICONS[agent]} {esc(agent)}</span>')
    return '<div class="fm-pipe">' + "".join(chips) + "</div>"


def line_html(ev: TraceEvent, new: bool) -> str:
    """One plain-language line (a headline) for the default view."""
    warn = ev.agent == "System" or ev.result_summary.startswith("Flagged")
    return (f'<div class="fm-line{" warn" if warn else ""}{" fm-new" if new else ""}" '
            f'style="--c:{AGENT_COLORS.get(ev.agent, "#9ca3af")}"><span>{AGENT_ICONS.get(ev.agent, "•")}</span>'
            f'<div><b>{esc(ev.agent)}</b> {esc(ev.result_summary)}</div></div>')


def event_html(ev: TraceEvent, new: bool) -> str:
    """One detailed event for the developer view."""
    color = AGENT_COLORS.get(ev.agent, "#9ca3af")
    is_llm = bool(ev.tool and ev.tool.startswith("llm"))
    tool = "LLM · " + ev.tool[5:-1] if is_llm and "(" in (ev.tool or "") else ev.tool
    warn = ("fallback" in ev.action or "rejected" in ev.action or "ignored" in ev.action
            or "FLAG" in ev.result_summary or ev.result_summary.startswith(("Failed", "FAILED")))
    ms = f'<span class="fm-ev-ms">{ev.duration_ms:,} ms</span>' if ev.duration_ms else ""
    body = f'<div class="fm-ev-body">{esc(ev.result_summary)}</div>' if ev.result_summary else ""
    tool_html = f'<span class="fm-ev-tool">{esc(tool)}</span>' if tool else ""
    details = ""
    for key, title in (("error_detail", "Technical details"), ("raw_output", "Full LLM output")):
        if ev.args.get(key):
            details += (f'<details class="fm-details"><summary>{title}</summary>'
                        f'<pre>{esc(ev.args[key])}</pre></details>')
    return (f'<div class="fm-ev{" warn" if warn else ""}{" fm-new" if new else ""}" style="--c:{color}">'
            f'<div class="fm-ev-head"><span>{AGENT_ICONS.get(ev.agent, "•")}</span>'
            f'<span class="fm-ev-agent">{esc(ev.agent)}</span><span>{esc(ev.action)}</span>{tool_html}{ms}</div>'
            f'{body}{details}</div>')


def active_html(step: StepStarted) -> str:
    verb = "is thinking" if (step.tool or "").startswith("llm") else "is working"
    return (f'<div class="fm-line fm-new" style="--c:{AGENT_COLORS.get(step.agent, "#9ca3af")}">'
            f'<span class="fm-dot"></span><div><b>{esc(step.agent)}</b> '
            f'<span class="fm-muted">{verb}…</span></div></div>')


def trace_html(events: list[TraceEvent], active: Optional[StepStarted] = None,
               dev: bool = False, animate: bool = False) -> str:
    """
    Default view: one plain sentence per agent. Developer view: every step.
    Only the newest line gets the fade-in class, so earlier lines never replay.
    """
    done = {e.agent for e in events if e.agent in PIPELINE}
    shown = [e for e in events if (e.kind != "headline") == dev]
    rows = [(event_html if dev else line_html)(e, animate and i == len(shown) - 1 and not active)
            for i, e in enumerate(shown)]
    if active:
        rows.append(active_html(active))
    if not rows:
        rows.append('<div class="fm-empty">Each agent will report here in one line as it finishes.</div>')
    return pipeline_html(done, active.agent if active else None) + "".join(rows)


def run_live(request: str, provider: str, slot, profile: Optional[dict] = None,
             on_partial: Optional[Callable[[PartialResult], None]] = None) -> AgentResult:
    """Run the agents, redraw the trace in `slot` after every step, and hand partial results on."""
    dev = st.session_state.get("dev", False)
    events: list[TraceEvent] = []
    for item in stream(request, provider=provider, profile=profile):
        if isinstance(item, StepStarted):
            slot.html(trace_html(events, active=item, dev=dev))
        elif isinstance(item, TraceEvent):
            events.append(item)
            slot.html(trace_html(events, dev=dev, animate=True))
        elif isinstance(item, PartialResult):
            if on_partial:
                on_partial(item)
        else:
            return item
    raise RuntimeError("agent run ended without a result")


def note_html(text: str, icon: str = "💬", color: str = ACCENT) -> str:
    return f'<div class="fm-note" style="--c:{color}">{icon} {esc(text)}</div>'


def reveal(slot, text: str, icon: str = "💬") -> None:
    """Show an already-verified explanation word by word (about half a second in total)."""
    words = text.split()
    step = max(1, len(words) // 12)
    for i in range(step, len(words), step):
        slot.html(note_html(" ".join(words[:i]), icon)[:-6] + '<span class="fm-cursor">▍</span></div>')
        time.sleep(0.04)
    slot.html(note_html(text, icon))


def writing(slot, what: str = "Writing the explanation") -> None:
    slot.html(f'<div class="fm-note"><span class="fm-dot" style="--c:{ACCENT};margin:0 6px 0 0"></span>'
              f'<span class="fm-muted">{esc(what)}…</span></div>')


# --------------------------------------------------------- decision card ----

def status_badge(status: str) -> str:
    color, icon = STATUS_STYLES.get(status, ("#9ca3af", "•"))
    return f'<span class="fm-badge" style="--c:{color}">{icon} {esc(status.title())}</span>'


def policy_chip(policy: str) -> str:
    return (f'<span class="fm-chip"><span style="color:{POLICY_COLORS.get(policy, "#9ca3af")}">●</span>'
            f'{esc(POLICY_LABELS.get(policy, policy))} policy</span>')


def bar_row(name: str, kind: str, level: int) -> str:
    return (f'<div class="fm-skill"><span>{esc(name)} <span class="fm-kind">{esc(kind)}</span></span>'
            f'<div class="fm-bar"><span style="width:{level / 3 * 100:.0f}%"></span></div>'
            f'<span class="fm-muted">{level}/3</span></div>')


def skill_bars(breakdown: dict, opp) -> str:
    """Per-skill bars from a record's {skill: points} breakdown (level = points / weight x 3)."""
    rows = []
    for skill, points in breakdown.items():
        if skill in BREAKDOWN_FIELDS:
            continue
        required = skill in opp.required_skills
        rows.append(bar_row(skill, "required" if required else "preferred",
                            round(points / (2 if required else 1) * 3)))
    return "".join(rows)


def score_text(data: dict) -> str:
    """'match 0.84 · skill fit 0.94 · 14.2 km from home'"""
    parts = []
    if data.get("score") is not None:
        parts.append(f"match {data['score']:.2f}")
    if data.get("skill_fit") is not None:
        parts.append(f"skill fit {data['skill_fit']:.2f}")
    if data.get("commute_km") is not None:
        parts.append(f"{data['commute_km']} km from home")
    return " · ".join(parts)


def candidate_block(sid: str, record: dict, opp, role: str) -> str:
    data = record["score_breakdown"].get(sid, {})
    past = next((s.past_opportunities for s in load_students() if s.id == sid), None)
    return (f'<div style="margin-bottom:var(--sp-3)"><div class="fm-person"><b>{esc(label(sid))}</b>'
            f'<span class="fm-muted fm-small">{esc(role)} · {esc(score_text(data))} · {past_text(past)}'
            f'</span></div>{skill_bars(data, opp)}</div>')


def decision_header(record: dict) -> str:
    opp = get_opportunity(record["opportunity_id"])
    when = (record.get("timestamp") or "")[:16].replace("T", " ")
    limit = f" · max {record['constraints']['max_km']:g} km commute" if record["constraints"].get("max_km") else ""
    return (f'<div style="display:flex;justify-content:space-between;gap:var(--sp-4);flex-wrap:wrap;'
            f'align-items:center"><div><div style="font-size:var(--fs-lg);font-weight:700">'
            f'{esc(opp.title)} <span class="fm-muted" style="font-weight:400">· {esc(record["id"])}</span></div>'
            f'<div class="fm-muted fm-small">{esc(opp.company)} · {esc(opp.area)} · {esc(opp.domain)} · '
            f'{plural(opp.slots, "slot")}{limit} · {esc(when)} UTC</div></div>'
            f'<div style="display:flex;gap:var(--sp-2)">{policy_chip(record["policy"])}'
            f'{status_badge(record["status"])}</div></div>')


def decision_body(record: dict) -> str:
    opp = get_opportunity(record["opportunity_id"])
    selected = "".join(candidate_block(sid, record, opp, "selected") for sid in record["selected"])
    if not selected:
        why = ("every eligible candidate is at the per-student cap"
               if record["constraints"].get("eligible_candidates") else "no eligible candidates")
        selected = f'<div class="fm-muted fm-small">Nobody selected: {esc(why)}.</div>'
    runner = (candidate_block(record["runner_up"], record, opp, "runner-up")
              if record.get("runner_up") and record["runner_up"] not in record["selected"] else
              '<div class="fm-muted fm-small">No runner-up.</div>')
    skipped = "".join(
        f'<div class="fm-note" style="--c:{WARN}"><b>{esc(label(sid))} skipped</b> '
        f'<span class="fm-muted">({esc(score_text(record["score_breakdown"].get(sid, {})))})</span>: '
        f'{esc(why)}.</div>' for sid, why in record["constraints"].get("skipped", {}).items())
    return (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));'
            f'gap:var(--sp-5);margin-top:var(--sp-4)"><div><div class="fm-label">Selected</div>{selected}</div>'
            f'<div><div class="fm-label">Runner-up</div>{runner}</div></div>{skipped}')


def appeal_html(record: dict) -> str:
    appeal = record.get("appeal")
    if not appeal:
        return ""
    color = "#9085e9" if appeal["outcome"] == "revised" else "#3987e5"
    return (f'<div class="fm-note" style="--c:{color}"><b>Appeal by {esc(label(appeal["student_id"]))}</b> '
            f'<span class="fm-muted">"{esc(appeal["text"])}"</span><br>'
            f'<b>Auditor: {esc(appeal["outcome"])}.</b> {esc(appeal["explanation"])}</div>')


# ------------------------------------------------------- recommendations ----

def show_recommendations(rec: dict, summary: Optional[str] = None):
    """
    Top matches (with per-skill bars) and the skills to learn next.
    Returns the placeholder for the explanation, so it can be revealed later.
    """
    slot = st.empty()
    if summary:
        slot.html(note_html(summary))
    else:
        writing(slot)

    total = len(load_opportunities())
    if rec["matches"]:
        st.html(f'<div class="fm-label" style="margin-top:var(--sp-4)">Your top {len(rec["matches"])} '
                f'listings · you qualify for {rec["eligible_count"]} of {total}</div>')
    for m in rec["matches"]:
        bars = "".join(bar_row(s, b["kind"], b["level"]) for s, b in m["breakdown"].items())
        stipend = f"₹{m['stipend']:,}/month" if m["stipend"] else "stipend not stated"
        commute = f" · {m['commute_km']} km from home" if m["commute_km"] is not None else ""
        color = DOMAIN_COLORS.get(m["domain"], ACCENT)
        st.html(f'<div class="fm-card" style="margin-bottom:var(--sp-2)"><div class="fm-match"><div>'
                f'<div style="font-weight:700">{esc(m["title"])}</div>'
                f'<div class="fm-muted fm-small" style="margin-bottom:var(--sp-2)">{esc(m["company"])} · '
                f'{esc(m["area"])}{esc(commute)} · <span style="color:{color}">●</span> {esc(m["domain"])} · '
                f'{esc(stipend)}</div>{bars}</div><div><div class="fm-score">{m["score"]:.2f}</div>'
                f'<div class="fm-muted fm-small" style="text-align:right">match<br>skill fit {m["skill_fit"]:.2f}'
                f'</div></div></div></div>')

    learn = rec["learn"]
    if learn["next"]:
        top = max(r["unlocks_in_interests"] for r in learn["next"])
        rows = "".join(
            f'<div class="fm-learn"><b>{esc(r["skill"])}</b><div>'
            f'<div class="fm-bar"><span style="width:{r["unlocks_in_interests"] / top * 100:.0f}%"></span></div>'
            f'<div class="fm-muted" style="font-size:var(--fs-xs);margin-top:var(--sp-1)">e.g. '
            f'{esc(", ".join(r["examples_in"]))}'
            + (f' · also opens {r["unlocks_elsewhere"]} elsewhere' if r["unlocks_elsewhere"] else "")
            + f'</div></div><span style="text-align:right">+{r["unlocks_in_interests"]}</span></div>'
            for r in learn["next"])
        st.html(f'<div class="fm-card" style="margin-top:var(--sp-3)"><div class="fm-label">Skills to learn next · '
                f'new listings in {esc(", ".join(learn["interests"]))} at beginner level</div>{rows}</div>')
    if learn["also_opens"]:
        chips = " ".join(f'<span class="fm-chip">{esc(r["skill"])} +{r["unlocks_elsewhere"]}</span>'
                         for r in learn["also_opens"])
        st.html(f'<div style="margin-top:var(--sp-3)"><div class="fm-label">Also opens, outside your interests'
                f'</div>{chips}</div>')
    return slot
