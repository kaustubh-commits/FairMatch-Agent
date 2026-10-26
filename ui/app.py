"""
FairMatch Bangalore: Streamlit entry point.

    streamlit run ui/app.py

This file only sets up the page, the sidebar and the navigation. Each page
lives in ui/views/ and is a normal Streamlit script. Pages are grouped by
who they are for: students or coordinators.
"""

import sys
from pathlib import Path

# Streamlit puts ui/ on the import path; we also need the project root
# so that `agents`, `tools` and `data_pipeline` can be imported.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="FairMatch Bangalore", page_icon="⚖️", layout="wide")

from ui.common import setup_page  # noqa: E402

pages = {
    "": [st.Page("views/home.py", title="Home", icon=":material/home:", default=True)],
    "For students": [
        st.Page("views/students.py", title="Find internships", icon=":material/school:", url_path="students"),
    ],
    "For coordinators": [
        st.Page("views/ask.py", title="Ask the agents", icon=":material/forum:", url_path="ask"),
        st.Page("views/policy_lab.py", title="Policy Lab", icon=":material/balance:", url_path="policy-lab"),
        st.Page("views/decision_log.py", title="Decision Log", icon=":material/fact_check:",
                url_path="decision-log"),
        st.Page("views/map.py", title="Map", icon=":material/map:", url_path="map"),
    ],
}
navigation = st.navigation(pages)
st.session_state.provider = setup_page()   # "gemini" or "demo-safe", read by the pages
navigation.run()
