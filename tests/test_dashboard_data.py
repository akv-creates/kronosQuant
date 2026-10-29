import numpy as np
import pandas as pd
import pytest

from kronosquant.dashboard_data import STRATEGIES, equity_and_drawdown, events_by_rule, load_table, scorecard, validation_candles
from kronosquant.paper import RiskLimits, run_paper
from kronosquant.splits import chronological_split


def candles(n=1000):
    close = 100 + np.arange(n, dtype=float)
    return pd.DataFrame({"timestamps": pd.date_range("2024-01-01", periods=n, freq="1h"), "open": close, "close": close, "high": close + 1, "low": close - 1, "volume": 1.0})


def test_validation_candles_return_exactly_the_validation_rows_and_never_test_rows():
    df = candles()
    val = chronological_split(len(df))["val"]
    test = chronological_split(len(df))["test"]
    out = validation_candles(df)
    assert len(out) == len(val)
    assert out["timestamps"].iloc[0] == df["timestamps"].iloc[val.start] and out["timestamps"].iloc[-1] == df["timestamps"].iloc[val.stop - 1]
    assert out["timestamps"].max() < df["timestamps"].iloc[test.start]  # nothing from the test split
    assert list(out.index) == list(range(len(out)))


def test_missing_result_files_are_skipped_not_errors(tmp_path):
    assert load_table(tmp_path / "nope.csv") is None
    sc = scorecard(tmp_path)
    assert sc.empty and list(sc.columns) == ["experiment", "signal", "estimate", "ci_lo", "ci_hi", "passed"]


def test_scorecard_reads_values_straight_from_the_csvs(tmp_path):
    pd.DataFrame({"signal": ["kronos", "reversal"], "dates": [100, 100], "mean_IC": [-0.0165, -0.003], "ci_lo99": [-0.10, -0.09], "ci_hi99": [0.07, 0.09],
                  "share_dates_pos": [0.45, 0.55], "PASS_primary": [False, False]}).to_csv(tmp_path / "stage10_cross_section.csv", index=False)
    pd.DataFrame({"baseline": ["hist_all"], "mean_err_kronos": [0.3], "mean_err_baseline": [0.32], "kronos_minus_baseline": [-0.012], "ci_lo99": [-0.05],
                  "ci_hi99": [0.027], "PASS": [False]}).to_csv(tmp_path / "stage11_primary.csv", index=False)
    sc = scorecard(tmp_path)
    assert len(sc) == 3
    k = sc[sc["signal"] == "kronos"].iloc[0]
    assert (k["estimate"], k["ci_lo"], k["ci_hi"], k["passed"]) == (-0.0165, -0.10, 0.07, False)
    v = sc[sc["signal"] == "vs hist_all"].iloc[0]
    assert v["estimate"] == pytest.approx(-0.012) and not v["passed"]


def test_scorecard_marks_a_pass_as_passed(tmp_path):
    pd.DataFrame({"signal": ["x"], "dates": [10], "mean_IC": [0.2], "ci_lo99": [0.05], "ci_hi99": [0.3], "share_dates_pos": [0.7], "PASS_primary": [True]}).to_csv(tmp_path / "stage10_cross_section.csv", index=False)
    assert scorecard(tmp_path)["passed"].tolist() == [True]


def test_drawdown_is_zero_at_peaks_and_negative_below():
    dd = equity_and_drawdown(pd.DataFrame({"equity": [1.0, 1.2, 0.9, 1.2, 1.5]}))
    np.testing.assert_allclose(dd["drawdown"], [0.0, 0.0, -0.25, 0.0, 0.0])
    assert dd["drawdown"].max() <= 0


def test_events_by_rule_counts_and_handles_empty():
    ev = pd.DataFrame({"bar": [1, 2, 3], "rule": ["max_position", "max_position", "kill_switch"]})
    out = events_by_rule(ev)
    assert out["rule"].tolist() == ["max_position", "kill_switch"] and out["count"].tolist() == [2, 1]
    assert events_by_rule(ev.iloc[0:0]).empty


def test_placeholder_strategies_run_through_the_paper_account():
    df = validation_candles(candles(2000))
    limits = RiskLimits(max_position=1.0, max_order=2.0, max_daily_loss=0.5, max_drawdown=0.5)
    for name, strat in STRATEGIES.items():
        r = run_paper(df, strat, limits=limits, fee_bps=10, slippage_bps=0)
        assert np.isfinite(r.metrics["net_return"]), name
    flat = run_paper(df, STRATEGIES["flat (does nothing)"], limits=limits, fee_bps=10, slippage_bps=0)
    assert flat.metrics["net_return"] == 0 and flat.metrics["total_cost"] == 0
