"""
For Students: pick your skills (or a student id) and see which real listings
fit you, plus which skill to learn next.

The form is already structured, so it hands the agents a ready profile:
Planner (records the profile) -> Matcher (recommend_for_profile,
skills_to_learn) -> Explainer (summary). All scores come from the same
score_match an allocator uses.
"""

import streamlit as st

from data_pipeline.constants import SKILLS
from tools.recommend import DEFAULT_HOURS
from tools.scoring import MIN_HOURS
from tools.store import load_students
from ui.common import page_header, run_live, show_recommendations, trace_html

LEVELS = ["None", "Beginner", "Intermediate", "Advanced"]

page_header("For students",
            "Tell us your skills and see which real Bangalore internships you qualify for, "
            "and which skill would open up the most new ones.")

provider = st.session_state.provider
form_col, result_col = st.columns([1, 1.5], gap="large")

with form_col:
    source = st.segmented_control("Profile", ["My skills", "Existing student ID"], default="My skills",
                                  key="student_source")
    profile = None
    if source == "Existing student ID":
        ids = [s.id for s in load_students()]
        sid = st.selectbox("Student ID", ids, index=ids.index("S011") if "S011" in ids else 0)
        student = next(s for s in load_students() if s.id == sid)
        known = ", ".join(f"{k} {v}/3" for k, v in student.skills.items() if v) or "none"
        st.caption(f"{student.branch}, year {student.year} · {student.hours_per_week} h/week · skills: {known}")
        profile = {"student_id": sid}
    else:
        with st.container(border=True):
            st.html('<div class="fm-muted fm-small">Your skill levels</div>')
            levels = {}
            left, right = st.columns(2)
            for i, skill in enumerate(SKILLS):
                col = left if i % 2 == 0 else right
                default = {"Python": "Intermediate", "Data analysis": "Intermediate"}.get(skill, "None")
                levels[skill] = col.select_slider(skill, LEVELS, value=default, key=f"lvl-{skill}")
            hours = st.slider("Hours available per week", 2, 40, DEFAULT_HOURS,
                              help=f"Listings need at least {MIN_HOURS} hours a week.")
        skills = {s: LEVELS.index(v) for s, v in levels.items() if v != "None"}
        profile = {"skills": skills, "hours": hours}

    go = st.button("Find my matches", type="primary", icon=":material/search:", width="stretch",
                   disabled=source != "Existing student ID" and not profile["skills"])
    st.html('<div style="font-weight:700;margin:.8rem 0 .4rem">Live agent trace</div>')
    trace_box = st.container(height=360, border=True)
    with trace_box:
        slot = st.empty()

cache = st.session_state.setdefault("student_results", {})
if go:
    result = run_live("Recommend internships for this student profile", provider, slot, profile=profile)
    cache["last"] = result
elif "last" in cache:
    slot.html(trace_html(cache["last"].events))
else:
    slot.html(trace_html([]))

with result_col:
    result = cache.get("last")
    if result is None:
        st.html('<div class="fm-card"><b>How it works</b><div class="fm-muted fm-small" style="margin-top:.35rem">'
                'Each listing needs certain skills (required ones count double). Your fit is the share of '
                'those points your levels earn, the same score allocators see. You qualify when you have at '
                f'least half the required skills and {MIN_HOURS}+ hours a week. "Skills to learn next" counts '
                'how many more listings you would qualify for at beginner level.</div></div>')
    elif result.data.get("recommend"):
        show_recommendations(result.data["recommend"], result.data["summary"])
    else:
        st.info(result.answer)
