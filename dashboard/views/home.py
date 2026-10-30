"""Home: what has been tested, and what the evidence says so far."""
import pandas as pd
import streamlit as st

import charts
import theme
from common import OUT, pct, period_text, show_chart, table_or_empty
from kronosquant.dashboard_data import GROUP_IC, GROUP_NET, GROUP_VOL, experiment_table, scorecard
from kronosquant.glossary import GLOSSARY
from kronosquant.metrics import paired_bootstrap_ci

CLASSICAL_EXPERIMENTS = {"exp_stage13_signal_screen"}  # tests of simple rules that do not involve Kronos
EQUITY_NAMES = {"kronos_longshort": "Kronos long/short", "momentum_longshort": "Momentum (12 bars)", "buy_hold": "Buy & Hold"}
EXPORT_HINT = ".venv/bin/python -I scripts/export_equity_curves.py"


def _summary_sentence(sc: pd.DataFrame) -> str:
    """One honest sentence derived from the scorecard (never hard-coded), so it cannot drift away from the results."""
    kronos = sc[~sc["exp_id"].isin(CLASSICAL_EXPERIMENTS)]
    classical = sc[sc["exp_id"].isin(CLASSICAL_EXPERIMENTS)]
    text = (f"{int(kronos['passed'].sum())} of {len(kronos)} comparisons in the Kronos experiments passed their pre-declared rule; "
            f"{int((kronos['status'] == 'Inconclusive').sum())} were inconclusive and {int((kronos['status'] == 'Failed').sum())} were reliably on the wrong side.")
    if len(classical) and classical["passed"].any():
        text += " The only pass is a simple classical momentum rule, and it is marginal and fragile (see its caveats under Experiments)."
    return text


def _kpis(exp: pd.DataFrame, sc: pd.DataFrame) -> None:
    total, done = len(exp), int((exp["state"] == "Completed").sum())
    passed, inconc, failed = (int((exp["outcome"] == k).sum()) for k in ("Passed", "Inconclusive", "Failed"))
    theme.kpi_row([
        theme.kpi("Total experiments", str(total), "rule declared first", help=GLOSSARY["Declared rule"]),
        theme.kpi("Completed", str(done), f"{done / total:.0%} of total" if total else ""),
        theme.kpi("Pending", str(total - done), "awaiting results", "warn" if total - done else ""),
        theme.kpi("Passed", str(passed), f"{int(sc['passed'].sum())} of {len(sc)} comparisons passed", "good" if passed else "", help=GLOSSARY["Passed / Failed / Inconclusive"]),
        theme.kpi("Inconclusive", str(inconc), f"{failed} reliably worse", help="The data could not tell skill from luck either way. This is not a hidden success."),
    ])


def _performance_card() -> None:
    theme.card_title("Strategy Performance (Net Returns)", help="Cumulative return after trading costs on the BTC validation period. A buy-and-hold line is shown for comparison.")
    eq = table_or_empty("equity_curves.csv", EXPORT_HINT)
    if eq is None:
        return
    eq = eq.assign(timestamp=pd.to_datetime(eq["timestamp"])).set_index("timestamp")
    colors = {"Kronos long/short": theme.PRIMARY, "Momentum (12 bars)": theme.SUCCESS, "Buy & Hold": charts.NEUTRAL}
    show_chart(charts.performance_figure({EQUITY_NAMES[c]: eq[c] for c in EQUITY_NAMES}, colors))
    ends = ", ".join(f"{EQUITY_NAMES[c]} {pct(eq[c].iloc[-1] - 1)}" for c in EQUITY_NAMES)
    theme.note(f"BTC 1h validation split, 10 bps one-way cost (assumed, not a verified Binance fee). Final net return: {ends}. Kronos uses cached forecasts and sits flat between 12-hour windows.")


def _recent_card(exp: pd.DataFrame) -> None:
    theme.card_title("Recent Experiments", help="Newest result files first. The Experiments page lists everything, with dates and headline numbers.")
    recent = exp.sort_values("date", ascending=False).head(7)
    st.markdown(theme.rows_table(["Experiment", "Status"], [[f"{r.stage} · {r.title}", r.outcome] for r in recent.itertuples()], badge_cols=(1,)), unsafe_allow_html=True)


def _forest_card(sc: pd.DataFrame) -> None:
    theme.card_title("Primary comparisons · 99% intervals", help=GLOSSARY["99% interval"])
    if sc.empty:
        theme.empty_state("No result files found in outputs/ yet, so there is nothing to compare.", ".venv/bin/python -I scripts/stage3_grid.py")
        return
    labels = {GROUP_IC: "IC (dot) with 99% interval; 0 = no skill", GROUP_VOL: "Error difference (dot) with 99% interval; below 0 = Kronos better", GROUP_NET: "Mean net return per period, bps"}
    for tab, group in zip(st.tabs(list(labels)), labels):
        with tab:
            part = sc[sc["group"] == group]
            if part.empty:
                st.caption("No results in this group yet.")
            else:
                show_chart(charts.forest_figure(part, labels[group]))
    theme.note("Filled dot = passed its declared rule; hollow = not passed. A bar that crosses the dashed line means the test could not tell skill from luck.")


def _distribution_card() -> None:
    theme.card_title("Distribution & Uncertainty", help="How the Kronos long/short result varies from one 12-hour window to the next, and the range the average could plausibly lie in.")
    wr = table_or_empty("kronos_window_returns.csv", EXPORT_HINT)
    if wr is None:
        return
    bps = wr["net_return"].to_numpy() * 1e4
    lo, hi = paired_bootstrap_ci(bps, level=0.99)
    show_chart(charts.distribution_figure(bps, float(bps.mean()), lo, hi, "Kronos long/short net return per 12-hour window (bps)"))
    theme.note(f"{len(bps)} windows, 10 bps one-way. The 99% interval for the mean is [{lo:+.1f}, {hi:+.1f}] bps.")


def render() -> None:
    theme.page_header("Research Overview", "What has been tested so far, and what the evidence says.", chip=f"Validation split · {period_text()}")
    exp, sc = experiment_table(OUT).rename(columns={"name": "title"}), scorecard(OUT)
    if exp.empty:
        theme.empty_state("No experiments are registered.")
        return
    if len(sc):
        st.info(_summary_sentence(sc) + " Research and paper trading only: nothing here is a trading recommendation.", icon=":material/info:")
    _kpis(exp, sc)
    st.write("")
    left, right = st.columns([1.7, 1], gap="medium")
    with left, st.container(border=True):
        _performance_card()
    with right, st.container(border=True):
        _recent_card(exp)
    st.write("")
    left, right = st.columns([1.5, 1], gap="medium")
    with left, st.container(border=True):
        _forest_card(sc)
    with right, st.container(border=True):
        _distribution_card()
    with st.expander("How to read these results"):
        for term in ("Information coefficient (IC)", "99% interval", "Passed / Failed / Inconclusive", "Net return"):
            st.markdown(f"**{term}.** {GLOSSARY[term]}")
