"""
Home: the entry point. Two paths, one line each, so a first-time visitor
knows where to go.
"""

import streamlit as st

st.html('<div class="fm-hero">Fair<span>Match</span> Bangalore</div>'
        '<div class="fm-lede">Matches engineering students to real Bangalore internships, checks that the '
        'same few students don\'t get every chance, and explains every decision.</div>')
st.html('<div style="height:var(--sp-5)"></div>')

student, coordinator = st.columns(2, gap="large")

with student:
    with st.container(border=True):
        st.html('<div class="fm-path"><div class="fm-path-title">🎓 I\'m a student</div>'
                '<div class="fm-muted">See which internships fit your skills, how far they are from home, '
                'and which skill would open the most new ones.</div></div>')
        if st.button("Find internships for me", type="primary", width="stretch", key="go-student"):
            st.switch_page("views/students.py")

with coordinator:
    with st.container(border=True):
        st.html('<div class="fm-path"><div class="fm-path-title">🧭 I\'m a coordinator</div>'
                '<div class="fm-muted">Rank candidates, compare allocation policies for fairness, and '
                'approve or review appeals, with every agent step visible.</div></div>')
        if st.button("Open the coordinator tools", width="stretch", key="go-coordinator"):
            st.switch_page("views/ask.py")
        a, b = st.columns(2)
        a.page_link("views/policy_lab.py", label="Policy Lab", icon=":material/balance:")
        b.page_link("views/decision_log.py", label="Decision Log", icon=":material/fact_check:")

st.html('<div class="fm-muted fm-small" style="margin-top:var(--sp-5)">Scores and decisions come from '
        'fixed, tested rules; the language model only reads requests and words explanations, and every '
        'number it writes is checked. A human approves every decision.</div>')
