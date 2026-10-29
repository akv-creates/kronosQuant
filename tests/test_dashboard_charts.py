import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")
from kronosquant.dashboard_data import buy_and_hold_equity, tripped_events, trade_markers  # noqa: E402
from kronosquant.paper import RiskLimits, run_paper  # noqa: E402

spec = importlib.util.spec_from_file_location("charts", Path(__file__).resolve().parents[1] / "dashboard" / "charts.py")
charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charts)


def candles(n=200):
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame({"timestamps": pd.date_range("2024-01-01", periods=n, freq="1h"), "open": open_, "close": close,
                         "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998, "volume": 1.0})


def test_trading_figure_has_the_expected_panels_and_marks_a_kill_switch():
    df = candles()
    limits = RiskLimits(max_position=1.0, max_order=1.0, max_daily_loss=0.5, max_drawdown=0.03)
    res = run_paper(df, lambda h, p: 1.0, limits=limits, fee_bps=5, slippage_bps=0)
    assert res.metrics["kill_switch"]  # a 3% limit on a 1%-per-bar random walk must trip
    fig = charts.trading_figure(df, res.frame, buy_and_hold_equity(df), trade_markers(res.frame, df), tripped_events(res.events), 0.03)
    kinds = [t.type for t in fig.data]
    assert kinds.count("candlestick") == 1 and kinds.count("scatter") >= 4  # buy markers, position, equity, benchmark, drawdown
    assert any("KILL SWITCH" in (a.text or "") for a in fig.layout.annotations)
    assert fig.layout.yaxis4.tickformat == ".0%"


def test_trading_figure_handles_no_trades_and_no_events():
    df = candles(50)
    limits = RiskLimits(max_position=1.0, max_order=1.0, max_daily_loss=0.5, max_drawdown=0.5)
    res = run_paper(df, lambda h, p: 0.0, limits=limits, fee_bps=5, slippage_bps=0)
    fig = charts.trading_figure(df, res.frame, buy_and_hold_equity(df), trade_markers(res.frame, df), tripped_events(res.events), 0.5)
    assert len([t for t in fig.data if getattr(t, "mode", None) == "markers"]) == 0  # no buy/sell markers when nothing traded


def test_forest_figure_marks_passes_with_a_filled_dot():
    rows = pd.DataFrame({"experiment": ["Stage 3 grid: IC", "Stage 10: x"], "signal": ["a", "b"], "estimate": [0.1, -0.02],
                         "ci_lo": [0.05, -0.1], "ci_hi": [0.2, 0.06], "passed": [True, False]})
    fig = charts.forest_figure(rows, "IC")
    assert list(fig.data[0].marker.symbol) == ["circle", "circle-open"]
    np.testing.assert_allclose(fig.data[0].error_x.array, [0.1, 0.08])
    np.testing.assert_allclose(fig.data[0].error_x.arrayminus, [0.05, 0.08])
