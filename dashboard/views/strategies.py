"""Strategies: run a placeholder strategy through the paper account and inspect it. Simulated fills only."""
import pandas as pd
import streamlit as st

import charts
import theme
from common import DATA, FETCH_HINT, num, paper_run, pct, show_chart, val_data
from kronosquant.backtest import sharpe_ratio
from kronosquant.dashboard_data import STRATEGIES, buy_and_hold_equity, equity_and_drawdown, events_by_rule, key_metrics, trade_markers, tripped_events
from kronosquant.glossary import GLOSSARY

WINDOWS = {"1W": 7, "1M": 30, "3M": 91, "All": None}
COSTS_BPS = [0, 5, 10, 20, 50]


def _config_card(symbols: list[str]) -> dict:
    """The control panel. Returns the chosen settings as a dict of keyword arguments for `paper_run`."""
    with st.container(border=True):
        theme.card_title("Configuration")
        c = st.columns(4, gap="medium")
        asset = c[0].selectbox("Asset (1h)", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0, help="Which market to replay. Only the validation period is used.")
        strategy = c[1].selectbox("Placeholder strategy", list(STRATEGIES), help="Simple rules that show how the paper account works. They are not research findings and not recommendations.")
        fee = c[2].number_input("Fee, bps one-way", 0.0, 100.0, 10.0, 1.0, help="Cost of each trade in basis points (1 bp = 0.01%). A round number, not a verified Binance fee.")
        slip = c[3].number_input("Slippage, bps one-way", 0.0, 100.0, 0.0, 1.0, help="Extra cost from filling slightly worse than the quoted price.")
        c = st.columns(4, gap="medium")
        max_pos = c[0].slider("Max position", 0.05, 1.0, 0.5, 0.05, help="The largest share of the account that can be invested at once (1.0 = fully invested).")
        max_order = c[1].slider("Max order size per bar", 0.05, 2.0, 0.25, 0.05, help="The most the position may change in a single bar. Forced exits are exempt.")
        max_day = c[2].slider("Max daily loss", 0.005, 0.5, 0.03, 0.005, format="%.3f", help="If the account loses this fraction in one day, trading halts until the next day.")
        max_dd = c[3].slider("Drawdown kill switch", 0.01, 0.9, 0.15, 0.01, help="If the account falls this far from its peak, trading stops for good.")
    return dict(asset=asset, strategy=strategy, fee=fee, slip=slip, max_pos=max_pos, max_order=max_order, max_day=max_day, max_dd=max_dd)


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


def render() -> None:
    theme.page_header("Strategy Investigation", "Run a placeholder strategy through the paper account and inspect it. Simulated fills only.", chip="Paper account · validation split", badge_label="Placeholder")
    files = sorted(DATA.glob("*_1h.csv"))
    if not files:
        theme.empty_state("No hourly market data is on this machine yet, so there is nothing to replay.", FETCH_HINT)
        return
    cfg = _config_card([f.name.replace("_1h.csv", "") for f in files])
    df = val_data(cfg["asset"])
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
