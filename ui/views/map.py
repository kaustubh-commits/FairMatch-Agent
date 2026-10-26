"""
Map page: where the opportunities are, and where students live.

  - dots   = opportunities, coloured by domain (hover for details)
  - glow   = heat layer of students' home areas
"""

import zlib

import pandas as pd
import pydeck as pdk
import streamlit as st

from data_pipeline.constants import BANGALORE_AREAS
from tools.store import data_source, load_opportunities, load_students
from ui.common import DOMAIN_COLORS, esc, page_header

page_header("Opportunity map",
            "Where the internships are, coloured by domain, and where students live.")


def hex_to_rgb(color: str) -> list[int]:
    return [int(color[i:i + 2], 16) for i in (1, 3, 5)]


def jitter(key: str, scale: float = 0.007) -> tuple[float, float]:
    """A small, stable offset so listings in the same area don't sit on one point."""
    h = zlib.crc32(key.encode())
    return ((h % 1000) / 1000 - 0.5) * scale, ((h // 1000 % 1000) / 1000 - 0.5) * scale


opportunities = load_opportunities()
students = load_students()

# --------------------------------------------------------------- controls ---
domains = [d for d in DOMAIN_COLORS if any(o.domain == d for o in opportunities)]
c1, c2 = st.columns([4, 1])
with c1:
    chosen = st.pills("Domains", domains, default=domains, selection_mode="multi",
                      label_visibility="collapsed")
with c2:
    show_heat = st.toggle("Student heat layer", value=True)

# ------------------------------------------------------------------ data ----
rows = []
for o in opportunities:
    if o.domain not in (chosen or []):
        continue
    dlat, dlon = jitter(o.id)
    rows.append({
        "lat": o.lat + dlat, "lon": o.lon + dlon, "color": hex_to_rgb(DOMAIN_COLORS[o.domain]),
        "title": o.title, "company": o.company, "area": o.area, "domain": o.domain,
        "stipend": f"₹{o.stipend:,}/month" if o.stipend else "Unpaid / not stated",
        "slots": o.slots, "skills": ", ".join(o.required_skills),
    })
opp_df = pd.DataFrame(rows)
student_df = pd.DataFrame([{"lat": s.lat, "lon": s.lon} for s in students])

layers = []
if show_heat:
    layers.append(pdk.Layer(
        "HeatmapLayer", student_df, get_position="[lon, lat]", radius_pixels=55,
        intensity=1.1, threshold=0.05, opacity=0.55,
        # one-hue ramp (teal), light = few students, dark->bright = many
        color_range=[[19, 78, 74], [17, 94, 89], [15, 118, 110], [13, 148, 136], [20, 184, 166], [45, 212, 191]],
    ))
if not opp_df.empty:
    layers.append(pdk.Layer(
        "ScatterplotLayer", opp_df, get_position="[lon, lat]", get_fill_color="color",
        get_radius=260, radius_min_pixels=5, radius_max_pixels=14,
        stroked=True, get_line_color=[11, 15, 25], line_width_min_pixels=2,  # surface ring
        pickable=True, auto_highlight=True, highlight_color=[255, 255, 255, 90],
    ))

deck = pdk.Deck(
    layers=layers,
    initial_view_state=pdk.ViewState(latitude=12.965, longitude=77.64, zoom=11.1, pitch=0),
    map_provider="carto", map_style=pdk.map_styles.CARTO_DARK,
    tooltip={"html": "<b>{title}</b><br/>{company}<br/>{area} · {domain}<br/>"
                     "Skills: {skills}<br/>{stipend} · {slots} slot(s)",
             "style": {"backgroundColor": "#131a2a", "color": "#e6e8ef", "fontSize": "12px",
                       "border": "1px solid #232b3f", "borderRadius": "8px", "padding": "8px 10px"}},
)

map_col, side_col = st.columns([3.2, 1], gap="large")
with map_col:
    st.pydeck_chart(deck, width="stretch", height=600)
with side_col:
    counts = {d: sum(1 for o in opportunities if o.domain == d) for d in domains}
    legend = "".join(
        f'<div style="display:flex;align-items:center;gap:.55rem;margin:.3rem 0;'
        f'opacity:{1 if d in (chosen or []) else .35}">'
        f'<span style="width:.8rem;height:.8rem;border-radius:50%;background:{DOMAIN_COLORS[d]};'
        f'box-shadow:0 0 0 2px #0b0f19"></span><span style="flex:1">{esc(d)}</span>'
        f'<span class="fm-muted">{counts[d]}</span></div>' for d in domains)
    st.html(f'<div class="fm-card"><div class="fm-muted fm-small" style="margin-bottom:.4rem">'
            f'OPPORTUNITIES BY DOMAIN</div>{legend}</div>')
    st.html('<div class="fm-card" style="margin-top:.8rem"><div class="fm-muted fm-small" '
            'style="margin-bottom:.4rem">STUDENT HOMES</div>'
            '<div style="height:.5rem;border-radius:999px;background:linear-gradient(90deg,#134e4a,#2dd4bf)">'
            '</div><div class="fm-muted fm-small" style="display:flex;justify-content:space-between;'
            f'margin-top:.25rem"><span>fewer</span><span>more</span></div>'
            f'<div class="fm-small" style="margin-top:.4rem">{len(students)} students across '
            f'{len(BANGALORE_AREAS)} areas</div></div>')

source = "Internshala listings" if data_source() == "kaggle" else "synthetic listings"
st.caption(f"Neighbourhoods are simulated: the {source} only say “Bangalore”, so each listing is placed in "
           "one of 15 tech areas with a fixed random seed (dots are spread slightly so they don't overlap). "
           "Student homes are synthetic. Map © CARTO, © OpenStreetMap contributors.")
