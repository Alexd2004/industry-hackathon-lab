"""Streamlit console. Run: streamlit run softsignal/app.py (no network needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # streamlit puts only this folder on the path

import streamlit as st

from softsignal.ui_loop import render_loop_tab
from softsignal.ui_results import render_results_tab

st.set_page_config(page_title="SoftSignal", layout="wide")
st.title("SoftSignal: teen-vs-adult likelihood")

results_tab, loop_tab = st.tabs(["Results", "Loop"])
with results_tab:
    render_results_tab()
with loop_tab:
    render_loop_tab()
