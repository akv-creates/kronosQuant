"""KronosQuant research dashboard (read-only). Run from the repository root:

    .venv/bin/python -I -m streamlit run dashboard/app.py

This file only sets up the page and the navigation. Each page lives in dashboard/views/, shared helpers in dashboard/common.py, styling in
dashboard/theme.py, figures in dashboard/charts.py, and all data logic (unit-tested) in src/kronosquant/dashboard_data.py.

The dashboard reads result files from outputs/ and re-runs the paper account on the VALIDATION split only. No model is run, nothing is traded,
and the test split is never loaded. Every number shown comes from a result file or a live paper-account run, never from a mock-up.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="KronosQuant", layout="wide", page_icon="▤")  # must be the first Streamlit call

import theme  # noqa: E402
from common import guarded  # noqa: E402
from views import datasets, experiments, home, models, reports, strategies  # noqa: E402

theme.inject_css()
st.logo(str(ROOT / "dashboard" / "assets" / "logo.svg"), size="large")
with st.sidebar:
    st.markdown('<div class="side-note">Research and paper trading only. Nothing here is a trading recommendation or investment advice.</div>', unsafe_allow_html=True)

PAGES = [
    ("Home", ":material/home:", "home", home, True),
    ("Experiments", ":material/science:", "experiments", experiments, False),
    ("Strategies", ":material/query_stats:", "strategies", strategies, False),
    ("Models", ":material/model_training:", "models", models, False),
    ("Datasets", ":material/database:", "datasets", datasets, False),
    ("Reports", ":material/description:", "reports", reports, False),
]
# The default page lives at the root URL, so it gets no url_path of its own.
st.navigation([st.Page(guarded(view.render), title=title, icon=icon, url_path=None if default else path, default=default) for title, icon, path, view, default in PAGES]).run()
