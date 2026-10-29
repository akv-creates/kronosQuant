"""Stage 8: read-only research dashboard. Run from the repo root:  .venv/bin/python -I -m streamlit run dashboard/app.py

Thin on purpose: all logic lives in kronosquant.dashboard_data (unit-tested). It reads result CSVs from outputs/ and re-runs the
paper account on the VALIDATION split only. No model is run, nothing is traded, and the test split is never loaded.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from kronosquant.dashboard_data import STRATEGIES, equity_and_drawdown, events_by_rule, load_table, scorecard, validation_candles  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.paper import RiskLimits, run_paper  # noqa: E402

OUT = ROOT / "outputs"
st.set_page_config(page_title="kronosQuant", layout="wide")
st.title("kronosQuant research dashboard")
st.warning("Research and paper trading only. Nothing here is financial advice, and a forecast is not a trading signal. "
           "Everything shown uses the validation split; the test split is untouched.")

tab_score, tab_tables, tab_paper = st.tabs(["Scorecard", "Result tables", "Paper account"])

with tab_score:
    sc = scorecard(OUT)
    if sc.empty:
        st.info("No result files found in outputs/. Run the experiment scripts first (outputs/ is git-ignored).")
    else:
        st.write("Each row is a pre-declared primary comparison. A row passes only if its 99% interval excludes the no-skill value "
                 "(IC above 0; or, for volatility, error difference below 0).")
        st.dataframe(sc.style.format({"estimate": "{:+.4f}", "ci_lo": "{:+.4f}", "ci_hi": "{:+.4f}"}), hide_index=True, use_container_width=True)
        st.metric("Primary comparisons passed", f"{int(sc['passed'].sum())} of {len(sc)}")
        for title, mask in [("Information coefficient (rank correlation, 0 = no skill)", ~sc["experiment"].str.startswith("Stage 11")),
                            ("Volatility error difference vs baseline (below 0 = Kronos better)", sc["experiment"].str.startswith("Stage 11"))]:
            part = sc[mask].reset_index(drop=True)
            if part.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 0.38 * len(part) + 1))
            y = range(len(part))
            ax.errorbar(part["estimate"], y, xerr=[part["estimate"] - part["ci_lo"], part["ci_hi"] - part["estimate"]], fmt="o", capsize=3)
            ax.axvline(0, color="grey", linestyle="--", linewidth=1)
            ax.set_yticks(list(y), [f"{e.split(':')[0]} | {s}" for e, s in zip(part["experiment"], part["signal"])], fontsize=8)
            ax.invert_yaxis()
            ax.set_title(title, fontsize=10)
            st.pyplot(fig)
            plt.close(fig)

with tab_tables:
    for label, name in [("Stage 3 grid (BTC 1h)", "grid_BTCUSDT_1h"), ("Stage 6 window strategies by cost", "stage6_strategies"),
                        ("Stage 9 fine-tuned vs pretrained", "stage9_evaluate_val"), ("Stage 10 cross-section", "stage10_cross_section"),
                        ("Stage 11 volatility, per asset", "stage11_per_asset")]:
        table = load_table(OUT / f"{name}.csv")
        st.subheader(label)
        if table is None:
            st.caption(f"outputs/{name}.csv not found on this machine.")
        else:
            st.dataframe(table, hide_index=True, use_container_width=True)

with tab_paper:
    st.write("Simulated fills only. A placeholder strategy trades the validation split through the risk layer; the strategy cannot see or change the limits.")
    files = sorted((ROOT / "data").glob("*_1h.csv"))
    if not files:
        st.info("No data/*_1h.csv found. Run scripts/fetch_binance.py first.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    symbols = [f.name.replace("_1h.csv", "") for f in files]
    asset = c1.selectbox("Asset", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
    strat_name = c1.selectbox("Placeholder strategy", list(STRATEGIES))
    fee = c2.number_input("Fee, bps one-way", 0.0, 100.0, 10.0, 1.0, help="Round number, not a verified Binance fee.")
    slip = c2.number_input("Slippage, bps one-way", 0.0, 100.0, 0.0, 1.0)
    max_pos = c3.slider("Max position", 0.05, 1.0, 0.5, 0.05)
    max_order = c3.slider("Max order size per bar", 0.05, 2.0, 0.25, 0.05)
    max_day = c1.slider("Max daily loss", 0.005, 0.5, 0.03, 0.005)
    max_dd = c2.slider("Drawdown kill switch", 0.01, 0.9, 0.15, 0.01)

    @st.cache_data(show_spinner=False)
    def val_data(symbol: str) -> pd.DataFrame:
        return validation_candles(load_candles(ROOT / "data" / f"{symbol}_1h.csv"))

    df = val_data(asset)
    res = run_paper(df, STRATEGIES[strat_name], limits=RiskLimits(max_position=max_pos, max_order=max_order, max_daily_loss=max_day, max_drawdown=max_dd),
                    fee_bps=fee, slippage_bps=slip)
    m = res.metrics
    st.caption(f"{asset} 1h validation: {len(df)} bars, {df['timestamps'].iloc[0]:%Y-%m-%d} to {df['timestamps'].iloc[-1]:%Y-%m-%d}")
    k = st.columns(5)
    k[0].metric("Net return", f"{m['net_return']:+.1%}")
    k[1].metric("Max drawdown", f"{m['max_drawdown']:.1%}")
    k[2].metric("Exposure", f"{m['exposure']:.0%}")
    k[3].metric("Total cost", f"{m['total_cost']:.2%}")
    k[4].metric("Kill switch", "TRIPPED" if m["kill_switch"] else "not tripped")
    ed = equity_and_drawdown(res.frame)
    st.line_chart(ed["equity"], height=240)
    st.line_chart(ed["drawdown"], height=160)
    st.subheader("Risk events")
    if res.events.empty:
        st.caption("No limit changed or blocked anything.")
    else:
        st.dataframe(events_by_rule(res.events), hide_index=True)
        st.dataframe(res.events.head(200), hide_index=True, use_container_width=True)
    st.caption("A halt flattens at the next open, so a loss can overshoot its limit. Strategies are placeholders: the numbers show how the mechanics behave, not skill.")
