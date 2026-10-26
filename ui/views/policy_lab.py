"""
Policy Lab: allocate every opportunity under all three policies and compare
fairness side by side. This is the key SDG 10 view.

The numbers come from the agents' compare flow (Matcher -> allocate,
Equity Analyst -> fairness_metrics + takeaway, Auditor -> rule check).
"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.common import POLICY_COLORS, POLICY_LABELS, esc, page_header, run_live, trace_html

page_header("Policy Lab", "SDG 10",
            "Same students, same opportunities, three allocation rules. Who ends up with the chances? "
            "(Reduced inequality.)")

REQUEST = "Compare the allocation policies and tell me which is fairest."
provider = st.session_state.provider
cache = st.session_state.setdefault("policy_lab", {})   # provider -> AgentResult

top_left, top_right = st.columns([3, 1])
with top_right:
    rerun = st.button("Run all 3 policies again", icon=":material/refresh:", width="stretch")

if rerun or provider not in cache:
    progress = st.empty()          # live trace while running, removed afterwards
    with progress.container():
        with st.status("Agents comparing the three policies…", expanded=True):
            cache[provider] = run_live(REQUEST, provider, st.empty())
    progress.empty()

result = cache[provider]
metrics, takeaway = result.data["metrics"], result.data["takeaway"]
source = "Gemini, checked against the numbers" if any(
    e.action == "takeaway (llm)" for e in result.events) else "template"

# ------------------------------------------------------------- takeaway ----
with top_left:
    st.html(f'<div class="fm-card" style="border-left:3px solid #fbbf24">'
            f'<div class="fm-small" style="color:#fbbf24;font-weight:700;margin-bottom:.25rem">'
            f'⚖️ Equity Analyst takeaway <span class="fm-muted" style="font-weight:400">· {esc(source)}</span></div>'
            f'<div style="font-size:1.02rem;line-height:1.45">{esc(takeaway)}</div></div>')

# ---------------------------------------------------------------- chart ----
PANELS = [  # (metric key, title, how to read it, formatter)
    ("avg_fit", "Average skill fit", "higher = better matches", lambda v: f"{v:.3f}"),
    ("gini", "Gini of opportunities", "lower = more equal", lambda v: f"{v:.3f}"),
    ("unique_recipients", "Unique recipients", "higher = more students reached", lambda v: f"{v:,}"),
    ("repeat_selection_rate", "Repeat selection", "lower = more first-timers", lambda v: f"{v:.1%}"),
]
policies = list(metrics)

fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.07,
                    subplot_titles=[f"<b>{t}</b><br><span style='font-size:11px;color:#9aa3b5'>{h}</span>"
                                    for _, t, h, _ in PANELS])
for col, (key, title, _, fmt) in enumerate(PANELS, start=1):
    for policy in policies:
        value = metrics[policy][key]
        fig.add_trace(go.Bar(
            x=[POLICY_LABELS[policy]], y=[value * 100 if key == "repeat_selection_rate" else value],
            name=POLICY_LABELS[policy], legendgroup=policy, showlegend=(col == 1),
            marker=dict(color=POLICY_COLORS[policy], cornerradius=4, line=dict(width=0)),
            text=[fmt(value)], textposition="outside", textfont=dict(color="#e6e8ef", size=12),
            hovertemplate=f"<b>{POLICY_LABELS[policy]}</b><br>{title}: {fmt(value)}<extra></extra>",
            cliponaxis=False,
        ), row=1, col=col)
    top = max(metrics[p][key] for p in policies) * (100 if key == "repeat_selection_rate" else 1)
    fig.update_yaxes(range=[0, top * 1.22 or 1], row=1, col=col,
                     ticksuffix="%" if key == "repeat_selection_rate" else "")

fig.update_layout(
    height=430, bargap=0.28, margin=dict(l=10, r=10, t=80, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#c9cfdc", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center",
                font=dict(color="#c9cfdc")),
    hoverlabel=dict(bgcolor="#131a2a", bordercolor="#232b3f", font=dict(color="#e6e8ef")),
)
fig.update_xaxes(showgrid=False, tickfont=dict(color="#9aa3b5", size=11), linecolor="#232b3f")
fig.update_yaxes(gridcolor="#1c2336", zeroline=False, tickfont=dict(color="#6b7489", size=10))
for annotation in fig.layout.annotations:
    annotation.update(font=dict(size=13, color="#e6e8ef"), yshift=8)

with st.container(border=True):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# ---------------------------------------------------- explanation + table ----
left, right = st.columns([1.3, 1], gap="large")
with left:
    st.markdown("**How to read this**")
    st.markdown(
        "- **Skill-first** gives every slot to the best-scoring student. Strong profiles often belong to "
        "students who were already selected before, so chances concentrate.\n"
        "- **Equal access** serves students with the fewest past opportunities first. It's the most equal, "
        "but matches are weaker.\n"
        "- **Hybrid** scales fit down by past opportunities (fit ÷ (1 + 0.5 × past)). It keeps most of the "
        "skill fit while sharply cutting repeat selection.\n"
        "- **Gini** measures how unevenly *all* opportunities (past + new) are spread across all students.")
with right:
    st.markdown("**Table view**")
    st.dataframe([{"Policy": POLICY_LABELS[p], "Avg fit": m["avg_fit"], "Gini": m["gini"],
                   "Unique": m["unique_recipients"], "Repeat %": round(m["repeat_selection_rate"] * 100, 1),
                   "Slots": m["slots_filled"]} for p, m in metrics.items()],
                 hide_index=True, width="stretch")
    audit = result.data["audit"]
    ok = all(not (a["over_cap"] or a["over_slots"]) for a in audit.values())
    st.caption("🛡️ Auditor: every allocation respects slot counts and the 2-per-student cap."
               if ok else f"🛡️ Auditor found rule violations: {audit}")

with st.expander(f"Agent trace for this comparison ({len(result.events)} steps)"):
    st.html(trace_html(result.events))
