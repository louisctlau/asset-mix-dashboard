"""Changelog — version history for Portfolio Lab."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Changelog — Portfolio Lab", layout="wide")
# Sidebar nav stays hidden; navigation is the sidebar link in app.py.
st.markdown("<style>[data-testid='stSidebarNav']{display:none;}</style>",
            unsafe_allow_html=True)

st.title("Changelog")
st.caption("Every shipped change to Portfolio Lab, newest first.")
st.page_link("app.py", label="← Back to dashboard")
st.divider()

md = (Path(__file__).resolve().parent.parent / "CHANGELOG.md").read_text()
st.markdown(md)
