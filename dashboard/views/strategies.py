"""Strategies: run a placeholder strategy through the paper account and inspect it. Simulated fills only."""
import pandas as pd
import streamlit as st

import charts
import theme
from common import DATA, HAS_LOCAL_DATA, OUT, num, paper_run, pct, show_chart, val_data
from kronosquant.backtest import sharpe_ratio
from kronosquant.dashboard_data import STRATEGIES, load_table, buy_and_hold_equity, equity_and_drawdown, events_by_rule, key_metrics, trade_markers, tripped_events
from kronosquant.data import DataError
from kronosquant.demo_data import MARKETS, MAX_UPLOAD_BYTES, MAX_UPLOAD_ROWS, parse_uploaded_candles, synthetic_candles
from kronosquant.glossary import GLOSSARY

WINDOWS = {"1W": 7, "1M": 30, "3M": 91, "All": None}
COSTS_BPS = [0, 5, 10, 20, 50]


def _config_card() -> dict:
    """The control panel. Returns the chosen settings as keyword arguments for `paper_run` (the candles are added by the caller)."""
    with st.container(border=True):
        theme.card_title("Configuration")
        c = st.columns(3, gap="medium")
        strategy = c[0].selectbox("Placeholder strategy", list(STRATEGIES), help="Simple rules that show how the paper account works. They are not research findings and not recommendations.")
        fee = c[1].number_input("Fee, bps one-way", 0.0, 100.0, 10.0, 1.0, help="Cost of each trade in basis points (1 bp = 0.01%). A round number, not a verified Binance fee.")
        slip = c[2].number_input("Slippage, bps one-way", 0.0, 100.0, 0.0, 1.0, help="Extra cost from filling slightly worse than the quoted price.")
        c = st.columns(4, gap="medium")
        max_pos = c[0].slider("Max position", 0.05, 1.0, 0.5, 0.05, help="The largest share of the account that can be invested at once (1.0 = fully invested).")
        max_order = c[1].slider("Max order size per bar", 0.05, 2.0, 0.25, 0.05, help="The most the position may change in a single bar. Forced exits are exempt.")
        max_day = c[2].slider("Max daily loss", 0.005, 0.5, 0.03, 0.005, format="%.3f", help="If the account loses this fraction in one day, trading halts until the next day.")
        max_dd = c[3].slider("Drawdown kill switch", 0.01, 0.9, 0.15, 0.01, help="If the account falls this far from its peak, trading stops for good.")
    return dict(strategy=strategy, fee=fee, slip=slip, max_pos=max_pos, max_order=max_order, max_day=max_day, max_dd=max_dd)


def _overview_tab(frame, bench, m, km, max_dd) -> None:
    ppy = m["periods_per_year"]
    b_ret = bench.pct_change().fillna(bench.iloc[0] - 1.0).to_numpy()
    b_sharpe = sharpe_ratio(b_ret, ppy)
    b_dd = float(equity_and_drawdown(pd.DataFrame({"equity": bench.to_numpy()}))["drawdown"].min())
    theme.kpi_row([
        theme.kpi("Net Return", pct(m["net_return"]), f"vs {pct(float(bench.iloc[-1] - 1))} buy & hold", help=GLOSSARY["Net return"]),
        theme.kpi("Sharpe Ratio", num(m["sharpe"]), f"vs {num(b_sharpe)} buy & hold", help=GLOSSARY["Sharpe ratio"]),
        theme.kpi("Max Drawdown", pct(m["max_drawdown"]), f"vs {pct(b_dd)} buy & hold; limit −{max_dd:.0%}", help=GLOSSARY["Drawdown"]),
    ])
    st.write("")
    left, right = st.columns([2.2, 1], gap="medium")
    with left, st.container(border=True):
        theme.card_title("Cumulative Returns")
        eq = frame["equity"].copy()
        eq.index = pd.to_datetime(eq.index)
        names = ("Strategy (net of costs)", "Buy & Hold (no costs)")
        show_chart(charts.performance_figure(dict(zip(names, (eq, bench))), dict(zip(names, (theme.PRIMARY, charts.NEUTRAL)))))
    with right, st.container(border=True):
        theme.card_title("Key Metrics")
        risk = "Kill switch tripped" if m["kill_switch"] else f"Daily halt, {m['bars_in_daily_halt']} bars" if m["bars_in_daily_halt"] else "No limit hit"
        rows = [("Turnover", f"{km['turnover']:.1f}"), ("Transaction costs", f"{km['cost_bps']:.0f} bps of equity"), ("Entries", str(km["entries"])),
                ("Avg. hold", "—" if pd.isna(km["avg_hold_bars"]) else f"{km['avg_hold_bars']:.0f} bars"),
                ("Hit rate (exposed bars)", "—" if pd.isna(km["hit_rate"]) else f"{km['hit_rate']:.1%}"), ("Exposure", f"{m['exposure']:.0%}"), ("Risk status", risk)]
        st.markdown(theme.kv_table(rows), unsafe_allow_html=True)


def _performance_tab(df, frame, bench, events, max_dd) -> None:
    window = st.segmented_control("Window", list(WINDOWS), default="All", label_visibility="collapsed") or "All"
    days = WINDOWS[window]
    start = df["timestamps"].iloc[-1] - pd.Timedelta(days=days) if days else df["timestamps"].iloc[0]
    keep = (df["timestamps"] >= start).to_numpy()
    markers = trade_markers(frame, df)
    markers = markers[markers["timestamp"] >= start]
    trips = tripped_events(events)
    trips = trips[trips["timestamp"] >= start] if len(trips) else trips
    with st.container(border=True):
        show_chart(charts.trading_figure(df[keep], frame[keep], bench[keep], markers, trips, max_dd))
    theme.note("Candles: hollow = up, filled = down. 1W, 1M and 3M show the last N days of the run (drag on the chart to zoom anywhere); the figures on the other tabs cover the whole run. "
               "A halt flattens at the next open, so a loss can overshoot its limit.")


def _metrics_tab(m) -> None:
    with st.container(border=True):
        theme.card_title("All metrics")
        rows = [("Net return", pct(m["net_return"])), ("Sharpe ratio (rf = 0)", num(m["sharpe"])), ("Max drawdown", pct(m["max_drawdown"])), ("Turnover", f"{m['turnover_total']:.2f}"),
                ("Total cost", f"{m['total_cost']:.2%} of equity"), ("Exposure", f"{m['exposure']:.1%}"), ("Kill switch", "tripped" if m["kill_switch"] else "not tripped"),
                ("Bars in daily halt", str(m["bars_in_daily_halt"])), ("Risk events", str(m["n_events"])), ("Periods per year", f"{m['periods_per_year']:.0f}")]
        st.markdown(theme.kv_table(rows), unsafe_allow_html=True)


def _robustness_tab(cfg, m) -> None:
    runs = [paper_run(**{**cfg, "fee": float(c)}) for c in COSTS_BPS]
    left, right = st.columns([1.2, 1], gap="medium")
    with left, st.container(border=True):
        theme.card_title("Cost sensitivity", help="The same strategy re-run with different trading costs. If the result only survives at very low costs, it is not robust.")
        show_chart(charts.cost_sensitivity_figure(COSTS_BPS, [r[2]["net_return"] for r in runs]))
    with right, st.container(border=True):
        theme.card_title("Table view")
        rows = [[f"{c} bps", pct(r[2]["net_return"]), pct(r[2]["max_drawdown"]), "tripped" if r[2]["kill_switch"] else "no"] for c, r in zip(COSTS_BPS, runs)]
        st.markdown(theme.rows_table(["One-way cost", "Net return", "Max drawdown", "Kill switch"], rows, num_cols=(1, 2)), unsafe_allow_html=True)
    loose = paper_run(**{**cfg, "max_pos": 1.0, "max_order": 2.0, "max_day": 0.99, "max_dd": 0.99})[2]
    with st.container(border=True):
        theme.card_title("Risk limits on versus effectively off")
        rows = [["Your limits", pct(m["net_return"]), pct(m["max_drawdown"]), f"{m['exposure']:.0%}"], ["Limits effectively off", pct(loose["net_return"]), pct(loose["max_drawdown"]), f"{loose['exposure']:.0%}"]]
        st.markdown(theme.rows_table(["Setting", "Net return", "Max drawdown", "Exposure"], rows, num_cols=(1, 2, 3)), unsafe_allow_html=True)
        theme.note("Smaller losses with limits usually mean the strategy stopped trading in a falling market, not that it found skill.")


def _details_tab(df, frame, events) -> None:
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        theme.card_title("Risk events", help="Moments when a limit changed or blocked an order.")
        if events.empty:
            st.caption("No limit changed or blocked anything.")
        else:
            by = events_by_rule(events)
            st.markdown(theme.rows_table(["Rule", "Count"], [[r.rule, str(r.count)] for r in by.itertuples()], num_cols=(1,)), unsafe_allow_html=True)
            st.dataframe(events.head(300), hide_index=True, use_container_width=True)
    with right, st.container(border=True):
        theme.card_title("Trade log")
        trades = trade_markers(frame, df)
        st.caption(f"{len(trades)} rebalances; each trades at that bar's open.")
        st.dataframe(trades.head(500), hide_index=True, use_container_width=True)


def _published_view() -> None:
    """Hosted version: the same placeholder strategies, precomputed once on the BTC validation split (scripts/publish_results.py). No simulation runs here."""
    ex = load_table(OUT / "paper_account_examples.csv")
    if ex is None:
        theme.empty_state("The precomputed paper-account results are not part of this deployment.")
        return
    st.info("These are precomputed paper-account runs on the real BTC validation data. The real Binance candles are not redistributed here, so to experiment interactively choose "
            "**Simulated market** or **Upload your own candles** above (or run the dashboard locally with the data downloaded; see the README).", icon=":material/info:")
    pick = st.selectbox("Placeholder strategy", [s for s in ex["strategy"].unique() if s in STRATEGIES], help="Simple rules that show how the paper account works. They are not research findings and not recommendations.")
    mine, bench = ex[(ex["strategy"] == pick) & (ex["scenario"] == "your limits")].set_index("fee_bps"), ex[ex["scenario"] == "benchmark"].iloc[0]
    default = mine.loc[10]
    off = ex[(ex["strategy"] == pick) & (ex["scenario"] == "limits effectively off")].iloc[0]
    theme.kpi_row([
        theme.kpi("Net Return", pct(default["net_return"]), f"vs {pct(bench['net_return'])} buy & hold", help=GLOSSARY["Net return"]),
        theme.kpi("Sharpe Ratio", num(default["sharpe"]), f"vs {num(bench['sharpe'])} buy & hold", help=GLOSSARY["Sharpe ratio"]),
        theme.kpi("Max Drawdown", pct(default["max_drawdown"]), f"vs {pct(bench['max_drawdown'])} buy & hold; limit −15%", help=GLOSSARY["Drawdown"]),
    ])
    theme.note("BTC validation split, 10 bps one-way fee, no slippage; limits: max position 0.50, max order 0.25 per bar, max daily loss 3%, drawdown kill switch 15%.")
    st.write("")
    left, right = st.columns([1.2, 1], gap="medium")
    with left, st.container(border=True):
        theme.card_title("Cost sensitivity", help="The same strategy re-run with different trading costs. If the result only survives at very low costs, it is not robust.")
        show_chart(charts.cost_sensitivity_figure([int(c) for c in mine.index], list(mine["net_return"])))
    with right, st.container(border=True):
        theme.card_title("Table view")
        rows = [[f"{int(c)} bps", pct(r.net_return), pct(r.max_drawdown), "tripped" if r.kill_switch else "no"] for c, r in mine.iterrows()]
        st.markdown(theme.rows_table(["One-way cost", "Net return", "Max drawdown", "Kill switch"], rows, num_cols=(1, 2)), unsafe_allow_html=True)
    with st.container(border=True):
        theme.card_title("Risk limits on versus effectively off")
        rows = [["Your limits", pct(default["net_return"]), pct(default["max_drawdown"]), f"{default['exposure']:.0%}"], ["Limits effectively off", pct(off["net_return"]), pct(off["max_drawdown"]), f"{off['exposure']:.0%}"]]
        st.markdown(theme.rows_table(["Setting", "Net return", "Max drawdown", "Exposure"], rows, num_cols=(1, 2, 3)), unsafe_allow_html=True)
        theme.note("Smaller losses with limits usually mean the strategy stopped trading in a falling market, not that it found skill.")


REAL, PRECOMPUTED, SYNTHETIC, UPLOAD = "Real data (validation split)", "Research runs (precomputed)", "Simulated market", "Upload your own candles"


def _synthetic_source() -> pd.DataFrame:
    with st.container(border=True):
        theme.card_title("Simulated market", help=GLOSSARY["Simulated market"])
        c = st.columns([1.7, 1, 1, 1], gap="medium")
        kind = c[0].selectbox("Market type", list(MARKETS), help="; ".join(f"{k}: {v}" for k, v in MARKETS.items()))
        n_bars = c[1].slider("Hourly bars", 500, 8000, 4000, 500, help="How long the simulated history is (4000 hours is about 5.5 months).")
        volatility = c[2].slider("Volatility per bar, %", 0.1, 2.0, 0.5, 0.1, help="Typical size of one hour's price move. 0.5% is roughly crypto-sized.") / 100
        seed = int(c[3].number_input("Random seed", 0, 9999, 0, 1, help="The same seed always gives the same market."))
        st.caption(MARKETS[kind])
    return synthetic_candles(kind, n_bars, seed, volatility)


def _upload_source() -> pd.DataFrame | None:
    with st.container(border=True):
        theme.card_title("Upload your own candles", help="Your file is processed in memory for this session only and is not stored.")
        st.caption(f"A CSV with columns time (timestamps, time, date or datetime), open, high, low, close and optionally volume. Regular bars (for example hourly) work best. Up to {MAX_UPLOAD_ROWS:,} rows and {MAX_UPLOAD_BYTES // 1_000_000} MB.")
        sample = synthetic_candles(n_bars=300, seed=1).to_csv(index=False)
        st.download_button("Download a sample file", sample, file_name="sample_candles.csv", mime="text/csv")
        uploaded = st.file_uploader("Candle file (CSV)", type=["csv"], label_visibility="collapsed")
        if uploaded is None:
            theme.empty_state("Choose a CSV file above to run the paper account on your own data.")
            return None
        try:
            candles = parse_uploaded_candles(uploaded.getvalue())
        except DataError as err:
            st.error(f"That file could not be used: {err}.")
            return None
        st.caption(f"Loaded {len(candles):,} bars from {candles['timestamps'].iloc[0]:%Y-%m-%d %H:%M} to {candles['timestamps'].iloc[-1]:%Y-%m-%d %H:%M}.")
    return candles


def _real_source() -> pd.DataFrame:
    symbols = [f.name.replace("_1h.csv", "") for f in sorted(DATA.glob("*_1h.csv"))]
    asset = st.selectbox("Asset (1h)", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0, help="Which market to replay. Only the validation period is used.")
    return val_data(asset)


def render() -> None:
    theme.page_header("Strategy Investigation", "Run a placeholder strategy through the paper account and inspect it. Simulated fills only.", chip="Paper account · simulated fills", badge_label="Placeholder")
    options = [REAL if HAS_LOCAL_DATA else PRECOMPUTED, SYNTHETIC, UPLOAD]
    source = st.segmented_control("Data source", options, default=options[0], key="strategy_source") or options[0]
    if source == PRECOMPUTED:
        if (OUT / "paper_account_examples.csv").exists():
            _published_view()
        else:
            theme.empty_state("No precomputed paper-account results are part of this deployment. Pick a simulated market or upload a file instead.")
        return
    if source == REAL:
        df = _real_source()
    else:
        st.warning(f"**Demo data.** {'Prices here are simulated.' if source == SYNTHETIC else 'This is your uploaded file.'} The results below illustrate how the paper account and its risk limits behave; "
                   "they are not part of the research results and say nothing about Kronos.", icon=":material/science:")
        df = _synthetic_source() if source == SYNTHETIC else _upload_source()
        if df is None:
            return
    cfg = {"candles": df, **_config_card()}
    frame, events, m = paper_run(**cfg)
    bench = buy_and_hold_equity(df)
    km = key_metrics(frame)
    st.write("")
    t_over, t_perf, t_met, t_rob, t_det = st.tabs(["Overview", "Performance", "Metrics", "Robustness", "Details"])
    with t_over:
        _overview_tab(frame, bench, m, km, cfg["max_dd"])
    with t_perf:
        _performance_tab(df, frame, bench, events, cfg["max_dd"])
    with t_met:
        _metrics_tab(m)
    with t_rob:
        _robustness_tab(cfg, m)
    with t_det:
        _details_tab(df, frame, events)
