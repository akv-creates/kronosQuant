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
    assert sc.empty and list(sc.columns) == ["exp_id", "experiment", "group", "signal", "estimate", "ci_lo", "ci_hi", "passed", "status", "lower_is_better"]


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


def _frame(positions):
    idx = pd.date_range("2024-01-01", periods=len(positions), freq="1h")
    return pd.DataFrame({"position": positions}, index=idx)


def test_trade_markers_found_at_the_open_of_each_change():
    from kronosquant.dashboard_data import trade_markers

    df = candles(6)
    f = _frame([0.0, 0.5, 1.0, 1.0, 0.0, -1.0])
    m = trade_markers(f, df)
    assert m["side"].tolist() == ["buy", "buy", "sell", "sell"]
    np.testing.assert_allclose(m["position_change"], [0.5, 0.5, -1.0, -1.0])
    np.testing.assert_allclose(m["price"], df["open"].iloc[[1, 2, 4, 5]])
    np.testing.assert_allclose(m["position_after"], [0.5, 1.0, 0.0, -1.0])
    assert m["timestamp"].tolist() == df["timestamps"].iloc[[1, 2, 4, 5]].tolist()
    assert trade_markers(_frame([0.0, 0.0, 0.0]), candles(3)).empty
    with pytest.raises(ValueError):
        trade_markers(_frame([0.0, 1.0]), candles(3))


def test_buy_and_hold_matches_the_paper_account_when_always_fully_long_and_free():
    from kronosquant.dashboard_data import buy_and_hold_equity

    df = candles(50)
    limits = RiskLimits(max_position=1.0, max_order=2.0, max_daily_loss=0.99, max_drawdown=0.99)
    bh = buy_and_hold_equity(df)
    assert bh.iloc[0] == pytest.approx(df["close"].iloc[0] / df["open"].iloc[0])
    r = run_paper(df, STRATEGIES["always long"], limits=limits, fee_bps=0, slippage_bps=0)
    # the account is flat in bar 0 (its first decision comes at bar 0's close), so compare growth from bar 1 on
    np.testing.assert_allclose(r.frame["equity"].to_numpy()[1:] / r.frame["equity"].iloc[1], bh.to_numpy()[1:] / bh.iloc[1], rtol=1e-9)


def test_tripped_events_keeps_only_trips():
    from kronosquant.dashboard_data import tripped_events

    ev = pd.DataFrame({"bar": [1, 2, 3, 4], "rule": ["max_position", "daily_loss_tripped", "daily_loss_halt", "kill_switch_tripped"]})
    assert tripped_events(ev)["rule"].tolist() == ["daily_loss_tripped", "kill_switch_tripped"]
    assert tripped_events(ev.iloc[0:0]).empty


def test_status_of_three_states_for_both_directions():
    from kronosquant.dashboard_data import status_of

    assert status_of(0.01, 0.2) == "Passed" and status_of(-0.2, -0.01) == "Failed" and status_of(-0.1, 0.1) == "Inconclusive"
    # lower-is-better (an error difference): below zero is good
    assert status_of(-0.2, -0.01, lower_is_better=True) == "Passed" and status_of(0.01, 0.2, lower_is_better=True) == "Failed"
    assert status_of(-0.05, 0.0, lower_is_better=True) == "Inconclusive"  # touching zero is not a pass


def test_scorecard_adds_stage6_and_stage12_rows_and_statuses(tmp_path):
    pd.DataFrame({"strategy": ["kronos_longshort", "kronos_longshort", "momentum_longshort"], "cost_bps": [0, 10, 10], "mean_window_bps": [-7.7, -20.5, -30.0],
                  "ci_lo99_bps": [-31.0, -43.3, -50.0], "ci_hi99_bps": [14.0, 1.4, -10.0]}).to_csv(tmp_path / "stage6_strategies.csv", index=False)
    pd.DataFrame({"setting": ["S5", "S20"], "IC": [0.01, 0.15], "IC_lo99": [-0.1, 0.02], "IC_hi99": [0.12, 0.28]}).to_csv(tmp_path / "stage12_samples.csv", index=False)
    sc = scorecard(tmp_path)
    s6 = sc[sc["exp_id"] == "exp_stage6_window_strategy"]
    assert len(s6) == 1 and s6.iloc[0]["estimate"] == pytest.approx(-20.5) and s6.iloc[0]["status"] == "Inconclusive"  # only the declared primary row (10 bps)
    s12 = sc[sc["exp_id"] == "exp_stage12_samples"].iloc[0]
    assert s12["signal"].startswith("S20") and s12["status"] == "Passed" and bool(s12["passed"])


def test_experiment_table_marks_missing_results_pending_and_outcomes(tmp_path):
    from kronosquant.dashboard_data import experiment_table

    pd.DataFrame({"variant": ["V0"], "IC": [0.1], "IC_lo99": [-0.05], "IC_hi99": [0.25], "PASS_primary": [False]}).to_csv(tmp_path / "grid_BTCUSDT_1h.csv", index=False)
    t = experiment_table(tmp_path).set_index("id")
    assert t.loc["exp_grid_btc1h", "state"] == "Completed" and t.loc["exp_grid_btc1h", "outcome"] == "Inconclusive"
    assert "+0.100" in t.loc["exp_grid_btc1h", "headline"]
    assert t.loc["exp_stage12_samples", "state"] == "Pending" and t.loc["exp_stage12_samples", "outcome"] == "Pending"
    assert (t["state"] == "Pending").sum() == len(t) - 1  # only the grid file exists


def test_dataset_inventory_reads_meta_and_csv(tmp_path):
    import json

    from kronosquant.dashboard_data import dataset_inventory

    pd.DataFrame({"timestamps": pd.date_range("2024-01-01", periods=5, freq="1D"), "open": 1.0}).to_csv(tmp_path / "XUSDT_1d.csv", index=False)
    (tmp_path / "XUSDT_1d.meta.json").write_text(json.dumps({"symbol": "XUSDT", "interval": "1d", "rows": 5, "gaps": 0, "fingerprint": "abcdef0123456789", "fetched_utc": "2026-10-29T10:00:00+00:00"}))
    inv = dataset_inventory(tmp_path)
    assert inv.iloc[0].to_dict() == {"symbol": "XUSDT", "interval": "1d", "rows": 5, "first": "2024-01-01", "last": "2024-01-05", "gaps": 0, "fingerprint": "abcdef012345", "fetched_utc": "2026-10-29"}
    assert dataset_inventory(tmp_path / "missing").empty


def test_code_version_marks_uncommitted_changes_and_unknown_outside_git(tmp_path):
    import shutil
    import subprocess

    from kronosquant.dashboard_data import code_version

    if shutil.which("git") is None:
        pytest.skip("git not installed")
    assert code_version(tmp_path) == "unknown"
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", *a], check=True, capture_output=True)
    run("init")
    (tmp_path / "f.txt").write_text("a")
    run("add", "f.txt")
    run("commit", "-m", "x")
    clean = code_version(tmp_path)
    assert len(clean) >= 7 and "uncommitted" not in clean
    (tmp_path / "f.txt").write_text("b")
    assert code_version(tmp_path) == clean + " (+ uncommitted changes)"


def test_declared_rule_returns_the_docstring_from_the_declaration_line(tmp_path):
    from kronosquant.dashboard_data import declared_rule

    f = tmp_path / "exp.py"
    f.write_text('"""Title line.\n\nDECLARED BEFORE RUNNING (do not edit):\n  primary: x > 0\n"""\nimport os\n')
    assert declared_rule(f).startswith("DECLARED BEFORE RUNNING") and "primary: x > 0" in declared_rule(f)
    g = tmp_path / "plain.py"
    g.write_text('"""Just a description."""\n')
    assert declared_rule(g) == "Just a description."


def test_pinned_revisions_read_from_source_text(tmp_path):
    from kronosquant.dashboard_data import pinned_revisions

    (tmp_path / "src" / "kronosquant").mkdir(parents=True)
    (tmp_path / "vendor").mkdir()
    (tmp_path / "src" / "kronosquant" / "forecast.py").write_text(f'TOKENIZER_REVISION = "{"a" * 40}"\nMODEL_REVISION = "{"b" * 40}"\n')
    (tmp_path / "vendor" / "KRONOS_COMMIT").write_text("c" * 40 + "\n")
    assert pinned_revisions(tmp_path) == {"tokenizer": "a" * 40, "model": "b" * 40, "kronos_commit": "c" * 40}
    assert pinned_revisions(tmp_path / "nowhere")["model"] == "unknown" if (tmp_path / "nowhere").exists() else True


def test_key_metrics_hand_computed():
    from kronosquant.dashboard_data import key_metrics

    f = pd.DataFrame({"position": [0.0, 1.0, 1.0, 0.0, 0.5, 0.5], "cost": [0.0, 0.001, 0.0, 0.001, 0.0005, 0.0],
                      "net_ret": [0.0, 0.01, -0.02, 0.0, 0.03, -0.01]})
    m = key_metrics(f)
    assert m["turnover"] == pytest.approx(2.5)  # 1 in, 1 out, 0.5 in
    assert m["cost_bps"] == pytest.approx(25.0)
    assert m["entries"] == 2 and m["avg_hold_bars"] == pytest.approx(4 / 2)  # exposed on bars 1,2,4,5: 4 bars over 2 entries
    assert m["hit_rate"] == pytest.approx(0.5)  # exposed returns: +1%, -2%, +3%, -1% -> 2 of 4
    flat = key_metrics(pd.DataFrame({"position": [0.0, 0.0], "cost": [0.0, 0.0], "net_ret": [0.0, 0.0]}))
    assert flat["entries"] == 0 and np.isnan(flat["avg_hold_bars"]) and np.isnan(flat["hit_rate"])


def test_scorecard_and_registry_include_the_stage13_primary_rows(tmp_path):
    from kronosquant.dashboard_data import EXPERIMENTS, experiment_table

    rows = [dict(candidate=c, primary=p, window=w, cost_bps=k, mean_bps=m, ci_lo99_bps=lo, ci_hi99_bps=hi) for c, p, w, k, m, lo, hi in [
        ("XSMOM_28", True, "primary", 10, 46.7, 1.0, 105.3), ("REV_5", True, "primary", 10, -47.0, -89.1, -11.6), ("TSMOM_28", True, "primary", 10, 66.7, -18.6, 162.1),
        ("XSMOM_28", True, "primary", 20, 41.1, -4.6, 99.7), ("TSMOM_14", False, "primary", 10, 70.8, -5.7, 151.9), ("XSMOM_28", True, "recent", 10, -1.1, -63.5, 56.7)]]
    pd.DataFrame(rows).to_csv(tmp_path / "stage13_screen.csv", index=False)
    pd.DataFrame({"candidate": ["XSMOM_28"], "year": [2019], "mean_bps": [78.6], "n_periods": [73]}).to_csv(tmp_path / "stage13_by_year.csv", index=False)
    sc = scorecard(tmp_path)
    s13 = sc[sc["exp_id"] == "exp_stage13_signal_screen"].set_index("signal")
    assert set(s13.index) == {"XSMOM_28", "REV_5", "TSMOM_28"}  # only primary candidates, primary window, 10 bps
    assert s13.loc["XSMOM_28", "status"] == "Passed" and s13.loc["REV_5", "status"] == "Failed" and s13.loc["TSMOM_28", "status"] == "Inconclusive"
    t = experiment_table(tmp_path).set_index("id")
    assert t.loc["exp_stage13_signal_screen", "outcome"] == "Passed" and "XSMOM_28" in t.loc["exp_stage13_signal_screen", "headline"]
    assert next(e for e in EXPERIMENTS if e["id"] == "exp_stage13_signal_screen")["caveat"]  # a pass is never shown without its caveats


def test_scorecard_reads_only_the_primary_paired_row_of_stage13b(tmp_path):
    rows = [dict(kind="arm", name="COMB", cost_bps=10, mean_bps=25.7, ci_lo99_bps=-30.2, ci_hi99_bps=82.7, primary=np.nan),
            dict(kind="paired_diff", name="COMB-BASE", cost_bps=0, mean_bps=22.0, ci_lo99_bps=-29.8, ci_hi99_bps=76.6, primary=False),
            dict(kind="paired_diff", name="COMB-BASE", cost_bps=10, mean_bps=18.8, ci_lo99_bps=-32.7, ci_hi99_bps=73.3, primary=True)]
    pd.DataFrame(rows).to_csv(tmp_path / "stage13b_incremental.csv", index=False)
    s = scorecard(tmp_path)
    s = s[s["exp_id"] == "exp_stage13b_incremental"]
    assert len(s) == 1 and s.iloc[0]["estimate"] == pytest.approx(18.8) and s.iloc[0]["status"] == "Inconclusive"


def test_code_version_marks_uncommitted_changes_only_when_asked(tmp_path):
    import subprocess

    from kronosquant.dashboard_data import code_version

    assert code_version(tmp_path, fallback="n/a") == "n/a"  # not a git repository
    run = lambda *a: subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    (tmp_path / "f.txt").write_text("x")
    run("add", ".")
    run("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "c")
    (tmp_path / "f.txt").write_text("changed")
    assert code_version(tmp_path).endswith("(+ uncommitted changes)")
    assert "uncommitted" not in code_version(tmp_path, mark_dirty=False)
