"""Experiments: every experiment with the outcome of its pre-declared rule."""
import streamlit as st

import theme
from common import OUT
from kronosquant.dashboard_data import experiment_table
from kronosquant.experiments import EXPERIMENTS
from kronosquant.glossary import GLOSSARY

FILTERS = ["All", "Completed", "Pending", "Passed", "Inconclusive", "Failed"]


def render() -> None:
    theme.page_header("Experiments", "Every experiment, declared before it ran, with its outcome.", chip="Outcomes use the pre-declared rule")
    exp = experiment_table(OUT).rename(columns={"name": "title"})
    choice = st.segmented_control("Show", FILTERS, default="All", label_visibility="collapsed") or "All"
    shown = exp
    if choice in ("Completed", "Pending"):
        shown = exp[exp["state"] == choice]
    elif choice != "All":
        shown = exp[exp["outcome"] == choice]
    with st.container(border=True):
        if shown.empty:
            st.caption(f"No experiments are {choice.lower()} right now.")
        else:
            rows = [[r.stage, r.title, r.model, r.outcome, r.date, r.headline] for r in shown.itertuples()]
            st.markdown(theme.rows_table(["Stage", "Experiment", "Model", "Status", "Generated", "Headline result (99% interval)"], rows, badge_cols=(3,), wrap_cols=(1, 2, 5)), unsafe_allow_html=True)
    caveats = {e["id"]: e["caveat"] for e in EXPERIMENTS if e.get("caveat")}
    for r in exp.itertuples():
        if r.outcome in ("Passed", "Inconclusive") and r.id in caveats:
            st.warning(f"**{r.stage} · {r.title}.** Read with care. {caveats[r.id]}", icon=":material/warning:")
    theme.note(f"{GLOSSARY['Passed / Failed / Inconclusive']} Exploratory means no pass/fail rule was declared; Pending means the results file has not been generated on this machine.")
