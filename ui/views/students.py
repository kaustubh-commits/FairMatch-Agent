"""
For Students: pick your skills (or a student id) and see which real listings
fit you, how far they are, and which skill to learn next.

The form is already structured, so it hands the agents a ready profile:
Planner (records the profile) -> Matcher (recommend_for_profile,
skills_to_learn) -> Explainer (summary). All scores come from the same
score_match an allocator uses.
"""

import streamlit as st

from data_pipeline.constants import BANGALORE_AREAS, DOMAINS, SKILLS
from tools.recommend import DEFAULT_HOURS
from tools.scoring import MIN_HOURS
from tools.store import load_students
from ui.common import dev_toggle, page_header, reveal, run_live, show_recommendations, trace_html

LEVELS = ["None", "Beginner", "Intermediate", "Advanced"]
NOT_SET = "Not specified"
GOALS = ["internship", "research", "hackathons", "leadership"]

page_header("Find internships",
            "Tell us your skills and see which real Bangalore internships you qualify for, "
            "and which skill would open up the most new ones.")

provider = st.session_state.provider
form_col, result_col = st.columns([1, 1.5], gap="large")

with form_col:
    source = st.segmented_control("Profile", ["My skills", "Existing student"], default="My skills",
                                  key="student_source", label_visibility="collapsed")
    if source == "Existing student":
        students = load_students()
        labels = {s.label: s for s in students}
        names = list(labels)
        default = next((i for i, n in enumerate(names) if n.endswith("(S011)")), 0)
        student = labels[st.selectbox("Student", names, index=default)]
        known = ", ".join(f"{k} {v}/3" for k, v in student.skills.items() if v) or "none"
        st.caption(f"Lives in {student.home_area} · {student.hours_per_week} hours a week · "
                   f"interested in {', '.join(student.interests)} · skills: {known}")
        profile = {"student_id": student.id}
    else:
        with st.container(border=True):
            st.html('<div class="fm-label">Your skills</div>')
            levels = {}
            left, right = st.columns(2)
            for i, skill in enumerate(SKILLS):
                col = left if i % 2 == 0 else right
                default = {"Python": "Intermediate", "Data analysis": "Intermediate"}.get(skill, "None")
                levels[skill] = col.select_slider(skill, LEVELS, value=default, key=f"lvl-{skill}")
        with st.container(border=True):
            st.html('<div class="fm-label">About you</div>')
            interests = st.multiselect("Interested in", DOMAINS, default=["Data", "ML"],
                                       help="Skills to learn are ranked by new listings in these areas.")
            a, b = st.columns(2)
            home = a.selectbox("Home area", [NOT_SET] + list(BANGALORE_AREAS), index=1 + list(BANGALORE_AREAS)
                               .index("Jayanagar"), help="Used to show how far each listing is.")
            goal = b.selectbox("Goal", GOALS)
            hours = st.slider("Hours a week you can give", 2, 40, DEFAULT_HOURS,
                              help=f"Listings need at least {MIN_HOURS} hours a week.")
        skills = {s: LEVELS.index(v) for s, v in levels.items() if v != "None"}
        profile = {"skills": skills, "hours": hours, "interests": interests, "goal": goal,
                   "home_area": None if home == NOT_SET else home}

    go = st.button("Find my matches", type="primary", icon=":material/search:", width="stretch",
                   disabled=source != "Existing student" and not profile["skills"])

with result_col:
    cache = st.session_state.setdefault("student_results", {})
    answer = st.container()
    head, toggle = st.columns([1.4, 1])
    head.html('<div style="font-weight:700;padding-top:6px">What the agents did</div>')
    with toggle:
        dev = dev_toggle("students")
    with st.container(border=True):
        slot = st.empty()

    if go:
        pending = {}

        def on_partial(partial) -> None:
            with answer:
                pending["slot"] = show_recommendations(partial.data["recommend"])

        result = run_live("Recommend internships for this student profile", provider, slot,
                          profile=profile, on_partial=on_partial)
        cache["last"] = result
        if "slot" in pending and result.data.get("recommend"):
            reveal(pending["slot"], result.data["summary"])
        else:
            with answer:
                st.info(result.answer)
    else:
        result = cache.get("last")
        slot.html(trace_html(result.events if result else [], dev=dev))
        with answer:
            if result is None:
                st.html('<div class="fm-card fm-muted fm-small">Your match score combines skill fit (required '
                        'skills count double) with how close the listing is to home. You qualify when you '
                        f'have at least half the required skills and {MIN_HOURS}+ hours a week.</div>')
            elif result.data.get("recommend"):
                show_recommendations(result.data["recommend"], result.data["summary"])
            else:
                st.info(result.answer)
