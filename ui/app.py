"""
FairMatch Bangalore: Streamlit entry point.

    streamlit run ui/app.py

This file only sets up the page, the sidebar and the navigation. Each page
lives in ui/views/ and is a normal Streamlit script.
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

pages = [
    st.Page("views/ask.py", title="Ask", icon=":material/forum:", url_path="ask", default=True),
    st.Page("views/students.py", title="For Students", icon=":material/school:", url_path="students"),
    st.Page("views/map.py", title="Map", icon=":material/map:", url_path="map"),
    st.Page("views/policy_lab.py", title="Policy Lab", icon=":material/balance:", url_path="policy-lab"),
    st.Page("views/decision_log.py", title="Decision Log", icon=":material/fact_check:",
            url_path="decision-log"),
]
navigation = st.navigation(pages)
st.session_state.provider = setup_page()   # "gemini" or "demo-safe", read by the pages
navigation.run()
