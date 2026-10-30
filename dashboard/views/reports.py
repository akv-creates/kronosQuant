"""Reports: everything needed to reproduce an experiment, plus a plain-language glossary."""
import pandas as pd
import streamlit as st

import theme
from common import DATA, OUT, ROOT, val_data
from kronosquant.dashboard_data import code_version, declared_rule, experiment_table
from kronosquant.experiments import EXPERIMENTS
from kronosquant.glossary import GLOSSARY


def _evaluation_period(experiment: dict) -> str:
    """The experiment's own period. Hourly experiments say 'validation', which is shown with the real BTC dates; daily ones state their own window."""
    if experiment["period"] != "validation":
        return experiment["period"]
    if not (DATA / "BTCUSDT_1h.csv").exists():
        return "validation split (data not on this machine)"
    v = val_data("BTCUSDT")
    return f"{v['timestamps'].iloc[0]:%Y-%m-%d} → {v['timestamps'].iloc[-1]:%Y-%m-%d} (validation split, BTC 1h)"


def render() -> None:
    theme.page_header("Reproducibility", "Full experiment details for transparency and verification.", chip="Reproduce, do not trust the dashboard")
    exp = experiment_table(OUT).set_index("id")
    names = {e["id"]: f"{e['stage']} · {e['name']}" for e in EXPERIMENTS}
    pick = st.selectbox("Experiment", list(names), format_func=names.get)
    e = next(x for x in EXPERIMENTS if x["id"] == pick)
    files = [OUT / f for f in e["results"]]
    generated = max(pd.Timestamp(f.stat().st_mtime, unit="s") for f in files).strftime("%Y-%m-%d %H:%M UTC") if all(f.exists() for f in files) else "not yet generated"
    left, right = st.columns([1.5, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown(f'<div class="card-title">Experiment Details &nbsp; {theme.badge(exp.loc[pick, "outcome"])}</div>', unsafe_allow_html=True)
        rows = [("Experiment ID", pick), ("Dataset", e["dataset"]), ("Code version", code_version(ROOT)), ("Model", e["model"]), ("Configuration", e["config"]), ("Random seed", e["seed"]),
                ("Evaluation period", _evaluation_period(e)), ("Cost assumptions", e["costs"]), ("Baselines", e["baselines"]), ("Generated", generated)]
        st.markdown(theme.kv_table(rows), unsafe_allow_html=True)
    with right, st.container(border=True):
        theme.card_title("Reproduce")
        st.code(e["command"], language="bash", wrap_lines=True)
        theme.note("Run from the repository root. Forecast results are cached, so a rerun reproduces the saved numbers; long runs resume where they stopped.")
        for f in files:
            if f.exists():
                st.download_button(f"Download {f.name}", f.read_bytes(), file_name=f.name, mime="text/csv", key=f"dl_{pick}_{f.name}")
    with st.container(border=True):
        theme.card_title("Rule declared before running", help=GLOSSARY["Declared rule"])
        script = ROOT / e["script"]
        st.code(declared_rule(script) if script.exists() else "The script file was not found in this checkout.", language="text")
    with st.expander("Glossary: what the terms mean"):
        for term, text in GLOSSARY.items():
            st.markdown(f"**{term}.** {text}")
