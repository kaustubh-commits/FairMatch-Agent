"""
Policy Lab: allocate every opportunity under all three policies and compare
fairness side by side, once and over several rounds.

The numbers come from the agents' compare flow (Matcher -> allocate + simulate,
Equity Analyst -> fairness_metrics + takeaway + outskirts check,
Auditor -> rule check).
"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from tools.fairness import OUTSKIRTS_FLAG_PTS, OUTSKIRTS_KM
from ui.common import (POLICY_COLORS, POLICY_LABELS, dev_toggle, esc, note_html, page_header,
                       run_live, trace_html)

page_header("Policy Lab",
            "Allocate every listing under three policies and compare who ends up with the chances.")

LIMITS = {"No limit": None, "15 km": 15, "10 km": 10}
provider = st.session_state.provider
cache = st.session_state.setdefault("policy_lab", {})   # (provider, limit) -> AgentResult

controls, rerun_col = st.columns([3, 1])
with controls:
    choice = st.segmented_control("Commute limit", list(LIMITS), default="No limit", key="commute_limit",
                                  help="Listings further than this from a student's home are not offered to them.")
limit = LIMITS[choice or "No limit"]
with rerun_col:
    rerun = st.button("Run again", icon=":material/refresh:", width="stretch")

request = "Compare the allocation policies" + (f" with a {limit} km commute limit" if limit else "")
key = (provider, limit)
if rerun or key not in cache:
    progress = st.empty()          # live trace while running, removed afterwards
    with progress.container():
        with st.status("The agents are comparing the three policies…", expanded=True):
            cache[key] = run_live(request, provider, st.empty())
    progress.empty()

result = cache[key]
metrics = result.data["metrics"]
policies = list(metrics)


def source_label(kind: str) -> str:
    return "written by Gemini, numbers checked" if kind == "llm" else "rule-based template"


def takeaway_card(text: str, kind: str) -> str:
    return (f'<div class="fm-card" style="border-left:3px solid #fbbf24">'
            f'<div class="fm-label" style="color:#fbbf24">⚖️ Equity Analyst · {esc(source_label(kind))}</div>'
            f'<div style="font-size:var(--fs-md);line-height:1.5">{esc(text)}</div></div>')


st.html(takeaway_card(result.data["takeaway"], result.data.get("takeaway_source", "template")))
flagged = result.data.get("outskirts_flags", [])
st.html(note_html(result.data["outskirts"], "📍", "#fbbf24" if flagged else "#34d399"))

# ---------------------------------------------------------------- chart ----
PANELS = [  # (metric key, title, how to read it, scale, formatter)
    ("avg_fit", "Average skill fit", "higher = better matches", 1, lambda v: f"{v:.3f}"),
    ("gini", "Gini of opportunities", "lower = more equal", 1, lambda v: f"{v:.3f}"),
    ("unique_recipients", "Students reached", "higher = more students get a chance", 1, lambda v: f"{v:,}"),
    ("repeat_selection_rate", "Repeat selection", "lower = more first-timers", 100, lambda v: f"{v:.0f}%"),
    ("avg_commute_km", "Average commute", "km from home, lower = closer", 1, lambda v: f"{v:.1f} km"),
    ("outskirts_gap_pts", "Outskirts selection gap",
     f"points; above {OUTSKIRTS_FLAG_PTS:g} = outskirts students left out", 1, lambda v: f"{v:+.1f}"),
]

fig = make_subplots(rows=2, cols=3, horizontal_spacing=0.08, vertical_spacing=0.24,
                    subplot_titles=[f"<b>{t}</b><br><span style='font-size:11px;color:#9aa3b5'>{h}</span>"
                                    for _, t, h, _, _ in PANELS])
for i, (key, title, _, scale, fmt) in enumerate(PANELS):
    row, col = i // 3 + 1, i % 3 + 1
    values = [(metrics[p][key] or 0) * scale for p in policies]
    for policy, value in zip(policies, values):
        fig.add_trace(go.Bar(
            x=[POLICY_LABELS[policy]], y=[value], name=POLICY_LABELS[policy], legendgroup=policy,
            showlegend=(i == 0), marker=dict(color=POLICY_COLORS[policy], cornerradius=4, line=dict(width=0)),
            text=[fmt(value)], textposition="outside", textfont=dict(color="#e6e8ef", size=12),
            hovertemplate=f"<b>{POLICY_LABELS[policy]}</b><br>{title}: {fmt(value)}<extra></extra>",
            cliponaxis=False,
        ), row=row, col=col)
    low = min(0, min(values) * 1.4)
    high = max(values) * 1.25 or 1
    if key == "outskirts_gap_pts":
        high = max(high, OUTSKIRTS_FLAG_PTS * 1.35)
        fig.add_hline(y=OUTSKIRTS_FLAG_PTS, line=dict(color="#fab219", width=1, dash="dot"), row=row, col=col,
                      annotation_text="flag", annotation_font=dict(color="#fab219", size=10),
                      annotation_position="top left")
    fig.update_yaxes(range=[low, high], row=row, col=col, ticksuffix="%" if scale == 100 else "")

fig.update_layout(
    height=640, bargap=0.28, margin=dict(l=10, r=10, t=70, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#c9cfdc", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.5, xanchor="center", font=dict(color="#c9cfdc")),
    hoverlabel=dict(bgcolor="#131a2a", bordercolor="#232b3f", font=dict(color="#e6e8ef")),
)
fig.update_xaxes(showgrid=False, tickfont=dict(color="#9aa3b5", size=11), linecolor="#232b3f")
fig.update_yaxes(gridcolor="#1c2336", zeroline=True, zerolinecolor="#2c3550",
                 tickfont=dict(color="#6b7489", size=10))
for annotation in fig.layout.annotations[:len(PANELS)]:
    annotation.update(font=dict(size=13, color="#e6e8ef"), yshift=6)

with st.container(border=True):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

a, b = st.columns(2, gap="large")
with a.expander("How to read these charts"):
    st.markdown(
        "- **Skill-first** gives each slot to the best match score (80% skill fit, 20% closeness to home). "
        "Strong profiles often belong to students who were selected before, so chances concentrate.\n"
        "- **Equal access** serves students with the fewest past opportunities first and ignores the commute. "
        "It's the most equal, but matches are weaker.\n"
        "- **Hybrid** uses 90% skill fit and 10% closeness, scaled down by past opportunities "
        "(÷ (1 + 0.5 × past opportunities)).\n"
        "- **Gini** measures how unevenly *all* opportunities (past + new) are spread across students.\n"
        f"- **Outskirts gap**: how much less often students living more than {OUTSKIRTS_KM:g} km from MG Road "
        "are picked. A commute penalty makes internships more practical but can leave them out, "
        f"so a gap above {OUTSKIRTS_FLAG_PTS:g} points is flagged.")
with b.expander("Table view"):
    st.dataframe([{"Policy": POLICY_LABELS[p], "Skill fit": m["avg_fit"], "Gini": m["gini"],
                   "Students reached": m["unique_recipients"],
                   "Repeat %": round(m["repeat_selection_rate"] * 100, 1),
                   "Avg commute (km)": m["avg_commute_km"], "Outskirts gap (pts)": m["outskirts_gap_pts"],
                   "Slots": m["slots_filled"]} for p, m in metrics.items()], hide_index=True, width="stretch")
    audit = result.data["audit"]
    ok = all(not (x["over_cap"] or x["over_slots"]) for x in audit.values())
    st.caption("Auditor: every allocation respects slot counts and the 2-per-student cap."
               if ok else f"Auditor found rule violations: {audit}")

# ------------------------------------------------- multi-round simulation ----
sims = result.data["simulation"]
rounds = [r["round"] for r in sims["hybrid"]]
st.html('<div class="fm-h2">What happens if the same policy is used round after round?</div>'
        '<div class="fm-caption">Each round allocates every listing again; everyone selected gains one '
        'past opportunity before the next round.</div>')
st.html(takeaway_card(result.data["rounds_takeaway"], result.data.get("rounds_takeaway_source", "template")))
st.html('<div style="height:var(--sp-3)"></div>')

LINE_PANELS = [  # (metric, title, how to read it, scale, tick suffix)
    ("gini", "Gini of opportunities", "lower = more equal", 1, ""),
    ("repeat_selection_rate", "Repeat selection", "share of slots won by the previous round's winners", 100, "%"),
]
line_fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.09,
                         subplot_titles=[f"<b>{t}</b><br><span style='font-size:11px;color:#9aa3b5'>{h}</span>"
                                         for _, t, h, _, _ in LINE_PANELS])
for col, (key, title, _, scale, suffix) in enumerate(LINE_PANELS, start=1):
    fmt = (lambda v: f"{v:.0f}%") if suffix else (lambda v: f"{v:.3f}")
    top = max(r[key] for p in policies for r in sims[p]) * scale
    ends = []   # (last value, policy) for the end-of-line labels
    for policy in policies:
        ys = [r[key] * scale for r in sims[policy]]
        ends.append((ys[-1], policy))
        line_fig.add_trace(go.Scatter(
            x=rounds, y=ys, mode="lines+markers", name=POLICY_LABELS[policy], legendgroup=policy,
            showlegend=(col == 1), line=dict(color=POLICY_COLORS[policy], width=2),
            marker=dict(size=8, color=POLICY_COLORS[policy], line=dict(color="#0b0f19", width=2)),
            hovertemplate=f"<b>{POLICY_LABELS[policy]}</b><br>Round %{{x}}<br>{title}: "
                          + ("%{y:.0f}%" if suffix else "%{y:.3f}") + "<extra></extra>",
        ), row=1, col=col)
    # Direct label at the end of each line; close values are nudged apart so they never overlap.
    shift, previous = 0, None
    for value, policy in sorted(ends):
        shift = shift + 14 if previous is not None and value - previous < top * 0.07 else 0
        line_fig.add_annotation(x=rounds[-1], y=value, text=fmt(value), showarrow=False, xshift=28,
                                yshift=shift, font=dict(size=11, color="#e6e8ef"), row=1, col=col)
        previous = value
    line_fig.update_yaxes(range=[0, top * 1.15 or 1], ticksuffix=suffix, row=1, col=col)

line_fig.update_layout(
    height=400, margin=dict(l=10, r=40, t=70, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#c9cfdc", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
    hoverlabel=dict(bgcolor="#131a2a", bordercolor="#232b3f", font=dict(color="#e6e8ef")),
    hovermode="x unified",
)
line_fig.update_xaxes(tickvals=rounds, ticktext=[f"Round {r}" for r in rounds], showgrid=False,
                      tickfont=dict(color="#9aa3b5", size=11), linecolor="#232b3f",
                      range=[0.7, len(rounds) + 0.45])
line_fig.update_yaxes(gridcolor="#1c2336", zeroline=False, tickfont=dict(color="#6b7489", size=10))
for annotation in line_fig.layout.annotations[:2]:
    annotation.update(font=dict(size=13, color="#e6e8ef"), yshift=6)

with st.container(border=True):
    st.plotly_chart(line_fig, width="stretch", config={"displayModeBar": False})

with st.expander("Round-by-round table"):
    st.dataframe([{"Policy": POLICY_LABELS[p], **{f"Round {r['round']}": f"Gini {r['gini']:.3f} · "
                                                   f"repeat {r['repeat_selection_rate']:.0%} · "
                                                   f"reached {r['cumulative_reach']}" for r in sims[p]}}
                  for p in policies], hide_index=True, width="stretch")
    st.caption("Reached = students selected at least once so far. In round 1, “previous winners” means "
               "students who already had a real past opportunity.")

with st.expander("What the agents did"):
    dev = dev_toggle("lab")
    st.html(trace_html(result.events, dev=dev))
