"""
Policy Lab: allocate every opportunity under all three policies and compare
fairness side by side, once and over several rounds.

The numbers come from the agents' compare flow (Matcher -> allocate,
Equity Analyst -> fairness_metrics + takeaway, Auditor -> rule check).
"""

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.common import POLICY_COLORS, POLICY_LABELS, esc, page_header, run_live, trace_html

page_header("Policy Lab",
            "Allocate every opportunity under three policies and compare who ends up with the chances.")

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
def source_label(kind: str) -> str:
    return "Gemini, checked against the numbers" if kind == "llm" else "template"


source = source_label(result.data.get("takeaway_source", "template"))

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

# ------------------------------------------------- multi-round simulation ----
sims = result.data["simulation"]
rounds = [r["round"] for r in sims["hybrid"]]
st.html('<div style="font-size:1.25rem;font-weight:700;margin:1.6rem 0 .2rem">What happens if the same policy '
        'is used round after round?</div><div class="fm-caption">Each round allocates every opportunity again; '
        'everyone selected gains +1 past opportunity before the next round.</div>')
st.html(f'<div class="fm-card" style="border-left:3px solid #fbbf24;margin-bottom:.8rem">'
        f'<div class="fm-small" style="color:#fbbf24;font-weight:700;margin-bottom:.25rem">'
        f'⚖️ Equity Analyst takeaway <span class="fm-muted" style="font-weight:400">· '
        f'{esc(source_label(result.data.get("rounds_takeaway_source", "template")))}</span></div>'
        f'<div style="font-size:1.02rem;line-height:1.45">{esc(result.data["rounds_takeaway"])}</div></div>')

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
    # Direct label at the end of each line (3 series, so every line gets one).
    # Labels whose values are close are nudged apart so they never overlap.
    shift, previous = 0, None
    for value, policy in sorted(ends):
        shift = shift + 14 if previous is not None and value - previous < top * 0.07 else 0
        line_fig.add_annotation(x=rounds[-1], y=value, text=fmt(value), showarrow=False, xshift=28,
                                yshift=shift, font=dict(size=11, color="#e6e8ef"),
                                row=1, col=col)
        previous = value
    line_fig.update_yaxes(range=[0, top * 1.15 or 1], ticksuffix=suffix, row=1, col=col)

line_fig.update_layout(
    height=400, margin=dict(l=10, r=40, t=80, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="sans-serif", color="#c9cfdc", size=12),
    legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0.5, xanchor="center"),
    hoverlabel=dict(bgcolor="#131a2a", bordercolor="#232b3f", font=dict(color="#e6e8ef")),
    hovermode="x unified",
)
line_fig.update_xaxes(tickvals=rounds, ticktext=[f"Round {r}" for r in rounds], showgrid=False,
                      tickfont=dict(color="#9aa3b5", size=11), linecolor="#232b3f", range=[0.7, len(rounds) + 0.45])
line_fig.update_yaxes(gridcolor="#1c2336", zeroline=False, tickfont=dict(color="#6b7489", size=10))
for annotation in line_fig.layout.annotations[:2]:
    annotation.update(font=dict(size=13, color="#e6e8ef"), yshift=8)

with st.container(border=True):
    st.plotly_chart(line_fig, width="stretch", config={"displayModeBar": False})

st.dataframe([{"Policy": POLICY_LABELS[p], **{f"Round {r['round']}": f"Gini {r['gini']:.3f} · "
                                               f"repeat {r['repeat_selection_rate']:.0%} · "
                                               f"reach {r['cumulative_reach']}" for r in sims[p]}}
              for p in policies], hide_index=True, width="stretch")
st.caption("Reach = students selected at least once so far. In round 1, “previous winners” means students who "
           "already had a real past opportunity.")

with st.expander(f"Agent trace for this comparison ({len(result.events)} steps)"):
    st.html(trace_html(result.events))
