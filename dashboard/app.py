"""Read-only research dashboard. Run from the repo root:  .venv/bin/python -I -m streamlit run dashboard/app.py

Thin on purpose: logic lives in kronosquant.dashboard_data (unit-tested), figures in dashboard/charts.py, styling in dashboard/theme.py.
It reads result files from outputs/ and re-runs the paper account on the VALIDATION split only. No model is run, nothing is traded,
and the test split is never loaded. Every number shown comes from a result file or a live paper-account run, never from a mock-up.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="KronosQuant", layout="wide", page_icon="▤")

import charts  # noqa: E402
import theme  # noqa: E402
from kronosquant.backtest import infer_periods_per_year, sharpe_ratio  # noqa: E402
from kronosquant.dashboard_data import (  # noqa: E402
    EXPERIMENTS, GROUP_IC, GROUP_NET, GROUP_VOL, STRATEGIES, buy_and_hold_equity, code_version, dataset_inventory, declared_rule, equity_and_drawdown,
    events_by_rule, experiment_table, key_metrics, load_table, pinned_revisions, scorecard, trade_markers, tripped_events, validation_candles)
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.metrics import paired_bootstrap_ci  # noqa: E402
from kronosquant.paper import RiskLimits, run_paper  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402

OUT, DATA = ROOT / "outputs", ROOT / "data"
theme.inject_css()
st.logo(str(ROOT / "dashboard" / "assets" / "logo.svg"), size="large")


@st.cache_data(show_spinner=False)
def val_data(symbol: str) -> pd.DataFrame:
    return validation_candles(load_candles(DATA / f"{symbol}_1h.csv"))


def period_text() -> str:
    if not (DATA / "BTCUSDT_1h.csv").exists():
        return "validation split"
    v = val_data("BTCUSDT")
    return f"{v['timestamps'].iloc[0]:%b %d, %Y} – {v['timestamps'].iloc[-1]:%b %d, %Y}"


def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:+.1%}".replace("-", "−")


def num(x: float, fmt: str = "{:.2f}") -> str:
    return "—" if pd.isna(x) else fmt.format(x).replace("-", "−")


@st.cache_data(show_spinner=False)
def paper_run(asset: str, strategy: str, fee: float, slip: float, max_pos: float, max_order: float, max_day: float, max_dd: float):
    df = val_data(asset)
    res = run_paper(df, STRATEGIES[strategy], limits=RiskLimits(max_position=max_pos, max_order=max_order, max_daily_loss=max_day, max_drawdown=max_dd), fee_bps=fee, slippage_bps=slip)
    return res.frame, res.events, res.metrics


# ---------------------------------------------------------------------------------------------------------- Home

def page_home():
    theme.page_header("Research Overview", "Track your experiments, compare models, and explore results.", chip=f"Validation split · {period_text()}")
    exp, sc = experiment_table(OUT), scorecard(OUT)
    total, done = len(exp), int((exp["state"] == "Completed").sum())
    passed, inconc, failed = (int((exp["outcome"] == k).sum()) for k in ("Passed", "Inconclusive", "Failed"))
    theme.kpi_row([
        theme.kpi("Total experiments", str(total), "pre-declared rules"),
        theme.kpi("Completed", str(done), f"{done / total:.0%} of total"),
        theme.kpi("Pending", str(total - done), "awaiting results", "warn" if total - done else ""),
        theme.kpi("Passed", str(passed), f"{int(sc['passed'].sum())} of {len(sc)} comparisons passed", "good" if passed else ""),
        theme.kpi("Inconclusive", str(inconc), f"{failed} reliably worse", ""),
    ])
    st.write("")
    left, right = st.columns([1.7, 1], gap="medium")
    with left, st.container(border=True):
        theme.card_title("Strategy Performance (Net Returns)")
        eq = load_table(OUT / "equity_curves.csv")
        if eq is None:
            st.info("outputs/equity_curves.csv not found. Run scripts/export_equity_curves.py.")
        else:
            eq = eq.assign(timestamp=pd.to_datetime(eq["timestamp"])).set_index("timestamp")
            names = {"kronos_longshort": "Kronos long/short", "momentum_longshort": "Momentum (12 bars)", "buy_hold": "Buy & Hold"}
            colors = {"Kronos long/short": theme.PRIMARY, "Momentum (12 bars)": theme.SUCCESS, "Buy & Hold": charts.NEUTRAL}
            st.plotly_chart(charts.performance_figure({names[c]: eq[c] for c in names}, colors), use_container_width=True, config={"displayModeBar": False})
            ends = ", ".join(f"{names[c]} {pct(eq[c].iloc[-1] - 1)}" for c in names)
            st.markdown(f'<div class="note">BTC 1h validation split, 10 bps one-way cost (assumed, not a verified Binance fee). Final net return: {ends}. '
                        "Kronos uses cached forecasts and sits flat between 12-hour windows.</div>", unsafe_allow_html=True)
    with right, st.container(border=True):
        theme.card_title("Recent Experiments")
        recent = exp.sort_values(["date"], ascending=False).head(7)
        st.markdown(theme.rows_table(["Experiment", "Status", "Date"], [[f"{r.stage} \u00b7 {r.name_}", r.outcome, r.date] for r in recent.rename(columns={"name": "name_"}).itertuples()],
                                     badge_cols=(1,)), unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.5, 1], gap="medium")
    with left, st.container(border=True):
        theme.card_title("Primary comparisons · 99% intervals")
        if sc.empty:
            st.info("No result files found in outputs/.")
        else:
            tabs = st.tabs([GROUP_IC, GROUP_VOL, GROUP_NET])
            for tab, group, label in zip(tabs, (GROUP_IC, GROUP_VOL, GROUP_NET), ("IC (dot) with 99% interval; 0 = no skill", "Error difference (dot) with 99% interval; below 0 = Kronos better", "Mean net return per window, bps")):
                with tab:
                    part = sc[sc["group"] == group]
                    if part.empty:
                        st.caption("No results yet.")
                    else:
                        st.plotly_chart(charts.forest_figure(part, label), use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="note">Filled dot = passed its declared rule; hollow = not passed. A bar that crosses the dashed line means the test could not tell skill from luck.</div>', unsafe_allow_html=True)
    with right, st.container(border=True):
        theme.card_title("Distribution & Uncertainty")
        wr = load_table(OUT / "kronos_window_returns.csv")
        if wr is None:
            st.info("outputs/kronos_window_returns.csv not found. Run scripts/export_equity_curves.py.")
        else:
            bps = wr["net_return"].to_numpy() * 1e4
            lo, hi = paired_bootstrap_ci(bps, level=0.99)
            st.plotly_chart(charts.distribution_figure(bps, float(bps.mean()), lo, hi, "Kronos long/short net return per 12-hour window (bps)"), use_container_width=True, config={"displayModeBar": False})
            st.markdown(f'<div class="note">{len(bps)} windows, 10 bps one-way. The 99% interval for the mean is [{lo:+.1f}, {hi:+.1f}] bps.</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------------------- Experiments

def page_experiments():
    theme.page_header("Experiments", "Every experiment, declared before it ran, with its outcome.", chip="Outcomes use the pre-declared rule")
    exp = experiment_table(OUT)
    choice = st.segmented_control("Show", ["All", "Completed", "Pending", "Passed", "Inconclusive", "Failed"], default="All", label_visibility="collapsed") or "All"
    if choice in ("Completed", "Pending"):
        exp = exp[exp["state"] == choice]
    elif choice != "All":
        exp = exp[exp["outcome"] == choice]
    with st.container(border=True):
        if exp.empty:
            st.caption("Nothing matches this filter.")
        else:
            st.markdown(theme.rows_table(["Stage", "Experiment", "Model", "Status", "Date", "Headline result (99% interval)"],
                                         [[r.stage, r.name_, r.model, r.outcome, r.date, r.headline] for r in exp.rename(columns={"name": "name_"}).itertuples()], badge_cols=(3,), wrap_cols=(1, 2, 5)), unsafe_allow_html=True)
    for r in experiment_table(OUT).itertuples():
        caveat = next((e.get("caveat") for e in EXPERIMENTS if e["id"] == r.id), None)
        if caveat and r.outcome == "Passed":
            st.warning(f"{r.stage} \u00b7 {r.name_ if hasattr(r, 'name_') else ''} Read with care. {caveat}")
    st.markdown('<div class="note">Status: Passed = the whole interval beat the no-skill value; Failed = reliably on the wrong side; Inconclusive = the interval includes the no-skill value, '
                "so the test could not tell; Exploratory = no declared pass/fail rule; Pending = results not yet available.</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------------ Strategies

def page_strategies():
    theme.page_header("Strategy Investigation", "Run a placeholder strategy through the paper account and inspect it. Simulated fills only.", chip="Paper account · validation split", badge_label="Placeholder")
    files = sorted(DATA.glob("*_1h.csv"))
    if not files:
        st.info("No data/*_1h.csv found. Run scripts/fetch_binance.py first.")
        return
    symbols = [f.name.replace("_1h.csv", "") for f in files]
    with st.container(border=True):
        theme.card_title("Configuration")
        c = st.columns(4, gap="medium")
        asset = c[0].selectbox("Asset (1h)", symbols, index=symbols.index("BTCUSDT") if "BTCUSDT" in symbols else 0)
        strat = c[1].selectbox("Placeholder strategy", list(STRATEGIES))
        fee = c[2].number_input("Fee, bps one-way", 0.0, 100.0, 10.0, 1.0, help="A round number, not a verified Binance fee.")
        slip = c[3].number_input("Slippage, bps one-way", 0.0, 100.0, 0.0, 1.0)
        c = st.columns(4, gap="medium")
        max_pos = c[0].slider("Max position", 0.05, 1.0, 0.5, 0.05)
        max_order = c[1].slider("Max order size per bar", 0.05, 2.0, 0.25, 0.05)
        max_day = c[2].slider("Max daily loss", 0.005, 0.5, 0.03, 0.005, format="%.3f")
        max_dd = c[3].slider("Drawdown kill switch", 0.01, 0.9, 0.15, 0.01)
    df = val_data(asset)
    frame, events, m = paper_run(asset, strat, fee, slip, max_pos, max_order, max_day, max_dd)
    bench = buy_and_hold_equity(df)
    ppy = m["periods_per_year"]
    b_ret = bench.pct_change().fillna(bench.iloc[0] - 1.0).to_numpy()
    b_sharpe, b_dd = sharpe_ratio(b_ret, ppy), float(equity_and_drawdown(pd.DataFrame({"equity": bench.to_numpy()}))["drawdown"].min())
    km = key_metrics(frame)
    st.write("")
    t_over, t_perf, t_met, t_rob, t_det = st.tabs(["Overview", "Performance", "Metrics", "Robustness", "Details"])

    with t_over:
        theme.kpi_row([
            theme.kpi("Net Return", pct(m["net_return"]), f"vs {pct(float(bench.iloc[-1] - 1))} buy & hold", "good" if m["net_return"] >= bench.iloc[-1] - 1 else "bad"),
            theme.kpi("Sharpe Ratio", num(m["sharpe"]), f"vs {num(b_sharpe)} buy & hold"),
            theme.kpi("Max Drawdown", pct(m["max_drawdown"]), f"vs {pct(b_dd)} buy & hold; limit −{max_dd:.0%}"),
        ])
        st.write("")
        left, right = st.columns([2.2, 1], gap="medium")
        with left, st.container(border=True):
            theme.card_title("Cumulative Returns")
            eq = frame["equity"].copy()
            eq.index = pd.to_datetime(eq.index)
            st.plotly_chart(charts.performance_figure({"Strategy (net of costs)": eq, "Buy & Hold (no costs)": bench}, {"Strategy (net of costs)": theme.PRIMARY, "Buy & Hold (no costs)": charts.NEUTRAL}),
                            use_container_width=True, config={"displayModeBar": False})
        with right, st.container(border=True):
            theme.card_title("Key Metrics")
            risk = "Kill switch tripped" if m["kill_switch"] else f"Daily halt, {m['bars_in_daily_halt']} bars" if m["bars_in_daily_halt"] else "No limit hit"
            st.markdown(theme.kv_table([("Turnover", f"{km['turnover']:.1f}"), ("Transaction costs", f"{km['cost_bps']:.0f} bps of equity"), ("Entries", str(km["entries"])),
                                        ("Avg. hold", "—" if pd.isna(km["avg_hold_bars"]) else f"{km['avg_hold_bars']:.0f} bars"), ("Hit rate (exposed bars)", "—" if pd.isna(km["hit_rate"]) else f"{km['hit_rate']:.1%}"),
                                        ("Exposure", f"{m['exposure']:.0%}"), ("Risk status", risk)]), unsafe_allow_html=True)

    with t_perf:
        window = st.segmented_control("Window", ["1W", "1M", "3M", "All"], default="All", label_visibility="collapsed") or "All"
        days = {"1W": 7, "1M": 30, "3M": 91, "All": None}[window]
        start = df["timestamps"].iloc[-1] - pd.Timedelta(days=days) if days else df["timestamps"].iloc[0]
        keep = (df["timestamps"] >= start).to_numpy()
        markers = trade_markers(frame, df)
        markers = markers[markers["timestamp"] >= start]
        trips = tripped_events(events)
        trips = trips[trips["timestamp"] >= start] if len(trips) else trips
        with st.container(border=True):
            st.plotly_chart(charts.trading_figure(df[keep], frame[keep], bench[keep], markers, trips, max_dd), use_container_width=True, config={"displayModeBar": False})
        st.markdown('<div class="note">Candles: hollow = up, filled = down. 1W, 1M and 3M show the last N days of the run (drag on the chart to zoom anywhere); the figures on the other tabs cover the whole run. '
                    "A halt flattens at the next open, so a loss can overshoot its limit.</div>", unsafe_allow_html=True)

    with t_met:
        with st.container(border=True):
            theme.card_title("All metrics")
            st.markdown(theme.kv_table([("Net return", pct(m["net_return"])), ("Sharpe ratio (rf = 0)", num(m["sharpe"])), ("Max drawdown", pct(m["max_drawdown"])), ("Turnover", f"{m['turnover_total']:.2f}"),
                                        ("Total cost", f"{m['total_cost']:.2%} of equity"), ("Exposure", f"{m['exposure']:.1%}"), ("Kill switch", "tripped" if m["kill_switch"] else "not tripped"),
                                        ("Bars in daily halt", str(m["bars_in_daily_halt"])), ("Risk events", str(m["n_events"])), ("Periods per year", f"{ppy:.0f}")]), unsafe_allow_html=True)

    with t_rob:
        costs = [0, 5, 10, 20, 50]
        runs = [paper_run(asset, strat, float(c), slip, max_pos, max_order, max_day, max_dd) for c in costs]
        left, right = st.columns([1.2, 1], gap="medium")
        with left, st.container(border=True):
            theme.card_title("Cost sensitivity")
            st.plotly_chart(charts.cost_sensitivity_figure(costs, [r[2]["net_return"] for r in runs]), use_container_width=True, config={"displayModeBar": False})
        with right, st.container(border=True):
            theme.card_title("Table view")
            st.markdown(theme.rows_table(["One-way cost", "Net return", "Max drawdown", "Kill switch"], [[f"{c} bps", pct(r[2]["net_return"]), pct(r[2]["max_drawdown"]), "tripped" if r[2]["kill_switch"] else "no"] for c, r in zip(costs, runs)],
                                         num_cols=(1, 2)), unsafe_allow_html=True)
        loose = paper_run(asset, strat, fee, slip, 1.0, 2.0, 0.99, 0.99)
        with st.container(border=True):
            theme.card_title("Risk limits on versus effectively off")
            st.markdown(theme.rows_table(["Setting", "Net return", "Max drawdown", "Exposure"], [["Your limits", pct(m["net_return"]), pct(m["max_drawdown"]), f"{m['exposure']:.0%}"],
                                                                                              ["Limits effectively off", pct(loose[2]["net_return"]), pct(loose[2]["max_drawdown"]), f"{loose[2]['exposure']:.0%}"]], num_cols=(1, 2, 3)), unsafe_allow_html=True)
            st.markdown('<div class="note">Smaller losses with limits usually mean the strategy stopped trading in a falling market, not that it found skill.</div>', unsafe_allow_html=True)

    with t_det:
        left, right = st.columns(2, gap="medium")
        with left, st.container(border=True):
            theme.card_title("Risk events")
            if events.empty:
                st.caption("No limit changed or blocked anything.")
            else:
                by = events_by_rule(events)
                st.markdown(theme.rows_table(["Rule", "Count"], [[r.rule, str(r.count)] for r in by.itertuples()], num_cols=(1,)), unsafe_allow_html=True)
                st.dataframe(events.head(300), hide_index=True, use_container_width=True)
        with right, st.container(border=True):
            theme.card_title("Trade log")
            allm = trade_markers(frame, df)
            st.caption(f"{len(allm)} rebalances; each trades at that bar's open.")
            st.dataframe(allm.head(500), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------------------------------------- Models

def page_models():
    theme.page_header("Models", "Kronos-small and its variants against the random walk. Forecast quality only, no costs.", chip="BTC 1h validation split")
    rev = pinned_revisions(ROOT)
    grid, s9 = load_table(OUT / "grid_BTCUSDT_1h.csv"), load_table(OUT / "stage9_evaluate_val.csv")
    left, right = st.columns([1, 1.5], gap="medium")
    with left, st.container(border=True):
        theme.card_title("Kronos-small")
        st.markdown(theme.kv_table([("Parameters", "24.7 million"), ("Context window", "512 candles"), ("Tokenizer", "NeoQuasar/Kronos-Tokenizer-base"), ("Tokenizer revision", rev["tokenizer"][:12]),
                                    ("Model revision", rev["model"][:12]), ("Upstream code commit", rev["kronos_commit"][:12]), ("Hardware", "CPU (Apple M1, 8 GB)"), ("Typical speed", "about 4 s per forecast (5 sampled paths)")]), unsafe_allow_html=True)
    with right, st.container(border=True):
        theme.card_title("Path error versus the random walk (12-bar horizon, lower is better)")
        if grid is None:
            st.info("outputs/grid_BTCUSDT_1h.csv not found.")
        else:
            same = grid[~grid["variant"].str.contains("horizon4")]
            labels, vals = list(same["variant"]), [float(v) for v in same["path_MAE"]]
            if s9 is not None:
                labels.append("fine-tuned")
                vals.append(float(s9.iloc[0]["path_MAE"]))
            st.plotly_chart(charts.comparison_bars(labels, vals, float(same.iloc[0]["rw_path_MAE"]), "random walk", "Mean absolute path error"), use_container_width=True, config={"displayModeBar": False})
    with st.container(border=True):
        theme.card_title("Comparison results")
        if grid is None:
            st.caption("No results yet.")
        else:
            rows = [[r.variant, f"{r.IC:+.3f}", f"[{r.IC_lo99:+.3f}, {r.IC_hi99:+.3f}]", f"{r.dir_acc:.1%}", f"{r.base_rate_up:.1%}", f"{r.path_MAE:.4f}", f"{r.rw_path_MAE:.4f}",
                     "Inconclusive" if r.IC_lo99 <= 0 <= r.IC_hi99 else "Passed" if r.IC_lo99 > 0 else "Failed"] for r in grid.itertuples()]
            if s9 is not None:
                r = s9.iloc[0]
                rows.append(["fine-tuned", f"{r['IC']:+.3f}", f"[{r['IC_lo99']:+.3f}, {r['IC_hi99']:+.3f}]", f"{r['dir_acc']:.1%}", f"{r['base_rate_up']:.1%}", f"{r['path_MAE']:.4f}", f"{r['rw_path_MAE']:.4f}",
                             "Inconclusive" if r["IC_lo99"] <= 0 <= r["IC_hi99"] else "Passed" if r["IC_lo99"] > 0 else "Failed"])
            st.markdown(theme.rows_table(["Setting", "IC", "99% interval", "Direction", "Base rate up", "Path error", "Random walk", "IC result"], rows, badge_cols=(7,), num_cols=(1, 3, 4, 5, 6)), unsafe_allow_html=True)
            st.markdown('<div class="note">V4 uses a 4-bar horizon, so its path error is on a different scale from the others. In every setting the path error was higher than the random walk, '
                        "and the 99% interval for the excess error excluded zero (see Result tables in the repository outputs).</div>", unsafe_allow_html=True)


# -------------------------------------------------------------------------------------------------------- Datasets

def page_datasets():
    theme.page_header("Datasets", "Downloaded market data and how it is split. Public Binance data, no keys.", chip="Binance public klines")
    inv = dataset_inventory(DATA)
    with st.container(border=True):
        theme.card_title("Chronological split (BTC 1h)")
        path = DATA / "BTCUSDT_1h.csv"
        if path.exists():
            ts = load_candles(path)["timestamps"]
            sp = chronological_split(len(ts))
            rows = [[name.capitalize(), f"{ts.iloc[r.start]:%Y-%m-%d}", f"{ts.iloc[r.stop - 1]:%Y-%m-%d}", f"{len(r):,}", "Untouched" if name == "test" else ""] for name, r in sp.items()]
            st.markdown(theme.rows_table(["Split", "From", "To", "Rows", "Status"], rows, badge_cols=(4,), num_cols=(3,)), unsafe_allow_html=True)
        st.markdown('<div class="note">Splits are consecutive and never shuffled. The test split is opened only after a validation pass, and only once. Known limits: one exchange, a hand-picked 12-asset universe (survivorship bias), '
                    "costs assumed rather than verified.</div>", unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        theme.card_title("Inventory")
        if inv.empty:
            st.info("No datasets found in data/. Run scripts/fetch_binance.py.")
        else:
            st.markdown(theme.rows_table(["Symbol", "Interval", "Rows", "First", "Last", "Gaps", "Fingerprint", "Fetched"], [[r.symbol, r.interval, f"{r.rows:,}", r.first, r.last, str(r.gaps), r.fingerprint, r.fetched_utc] for r in inv.itertuples()],
                                         num_cols=(2, 5)), unsafe_allow_html=True)


# --------------------------------------------------------------------------------------------------------- Reports

def page_reports():
    theme.page_header("Reproducibility", "Full experiment details for transparency and verification.", chip="Reproduce, do not trust the dashboard")
    exp = experiment_table(OUT).set_index("id")
    names = {e["id"]: f"{e['stage']} · {e['name']}" for e in EXPERIMENTS}
    pick = st.selectbox("Experiment", list(names), format_func=names.get)
    e = next(x for x in EXPERIMENTS if x["id"] == pick)
    row = exp.loc[pick]
    files = [OUT / f for f in e["results"]]
    period = "—"
    if (DATA / "BTCUSDT_1h.csv").exists():
        v = val_data("BTCUSDT")
        period = f"{v['timestamps'].iloc[0]:%Y-%m-%d} → {v['timestamps'].iloc[-1]:%Y-%m-%d} (validation split)"
    generated = max(pd.Timestamp(f.stat().st_mtime, unit="s") for f in files).strftime("%Y-%m-%d %H:%M UTC") if all(f.exists() for f in files) else "not yet generated"
    left, right = st.columns([1.5, 1], gap="medium")
    with left, st.container(border=True):
        st.markdown(f'<div class="card-title">Experiment Details &nbsp; {theme.badge(row["outcome"])}</div>', unsafe_allow_html=True)
        st.markdown(theme.kv_table([("Experiment ID", pick), ("Dataset", e["dataset"]), ("Code version", code_version(ROOT)), ("Model", e["model"]), ("Configuration", e["config"]), ("Random seed", e["seed"]),
                                    ("Evaluation period", period), ("Cost assumptions", e["costs"]), ("Baselines", e["baselines"]), ("Generated", generated)]), unsafe_allow_html=True)
    with right, st.container(border=True):
        theme.card_title("Reproduce")
        st.code(e["command"], language="bash")
        st.markdown('<div class="note">Run from the repository root. Forecast results are cached, so a rerun reproduces the saved numbers; long runs resume where they stopped.</div>', unsafe_allow_html=True)
        for f in files:
            if f.exists():
                st.download_button(f"Download {f.name}", f.read_bytes(), file_name=f.name, mime="text/csv", key=f"dl_{pick}_{f.name}")
    with st.container(border=True):
        theme.card_title("Rule declared before running")
        script = ROOT / e["script"]
        st.code(declared_rule(script) if script.exists() else "script not found", language="text")


pages = [
    st.Page(page_home, title="Home", icon=":material/home:", url_path="home", default=True),
    st.Page(page_experiments, title="Experiments", icon=":material/science:", url_path="experiments"),
    st.Page(page_strategies, title="Strategies", icon=":material/query_stats:", url_path="strategies"),
    st.Page(page_models, title="Models", icon=":material/model_training:", url_path="models"),
    st.Page(page_datasets, title="Datasets", icon=":material/database:", url_path="datasets"),
    st.Page(page_reports, title="Reports", icon=":material/description:", url_path="reports"),
]
st.navigation(pages).run()
