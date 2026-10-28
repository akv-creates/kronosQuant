import numpy as np
import pandas as pd
import pytest

from kronosquant.baselines import BASELINES, drift, momentum, random_walk
from kronosquant.data import has_gap
from kronosquant.metrics import (
    cumulative_log_returns,
    directional_accuracy,
    end_mae,
    information_coefficient,
    paired_bootstrap_ci,
    path_mae,
)


def test_cumulative_log_returns_hand_computed():
    out = cumulative_log_returns(100.0, np.array([110.0, 100.0]))
    np.testing.assert_allclose(out, [np.log(1.1), 0.0])


def test_mae_hand_computed():
    pred = np.array([[0.0, 0.0], [1.0, 1.0]])
    actual = np.array([[1.0, 2.0], [1.0, 0.0]])
    assert path_mae(pred, actual) == pytest.approx((1 + 2 + 0 + 1) / 4)
    assert end_mae(pred, actual) == pytest.approx((2 + 1) / 2)


def test_directional_accuracy_skips_zero_predictions():
    acc, n = directional_accuracy(np.array([1.0, -1.0, 0.0, 2.0]), np.array([0.5, 0.5, 1.0, 3.0]))
    assert n == 3 and acc == pytest.approx(2 / 3)
    acc, n = directional_accuracy(np.zeros(3), np.ones(3))
    assert n == 0 and np.isnan(acc)


def test_information_coefficient_perfect_inverse_and_constant():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert information_coefficient(x, x * 10) == pytest.approx(1.0)
    assert information_coefficient(x, -x) == pytest.approx(-1.0)
    assert np.isnan(information_coefficient(np.zeros(4), x))


def test_baselines_hand_computed():
    close = np.array([100.0, 100.0, 110.0, 121.0])  # +10% each of the last two steps
    horizon = 2
    np.testing.assert_array_equal(random_walk(close, horizon), [0.0, 0.0])
    step = np.diff(np.log(close)).mean()
    np.testing.assert_allclose(drift(close, horizon), [step, 2 * step])
    past = np.log(121 / 100)  # last 2 candles: 100 -> 121
    np.testing.assert_allclose(momentum(close, horizon), [past / 2, past])


def test_baselines_are_deterministic_and_use_only_history():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, 400)))
    for fn in BASELINES.values():
        np.testing.assert_array_equal(fn(close.copy(), 12), fn(close.copy(), 12))
        assert fn(close, 12).shape == (12,)


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    diff = np.random.default_rng(0).normal(0.5, 1.0, 200)
    lo, hi = paired_bootstrap_ci(diff, seed=3)
    assert lo < diff.mean() < hi
    assert (lo, hi) == paired_bootstrap_ci(diff, seed=3)
    z_lo, z_hi = paired_bootstrap_ci(np.random.default_rng(1).normal(0, 1, 200))
    assert z_lo < 0 < z_hi  # no true effect -> interval should straddle zero


def test_has_gap_detects_missing_bars_inside_range_only():
    ts = pd.date_range("2024-01-01 09:35", periods=6, freq="5min").to_series().reset_index(drop=True)
    ts.iloc[3:] += pd.Timedelta(minutes=55)
    df = pd.DataFrame({"timestamps": ts})
    assert has_gap(df, range(0, 6))
    assert not has_gap(df, range(0, 3))  # gap sits between rows 2 and 3
    assert has_gap(df, range(2, 4))


def test_select_windows_skips_gaps_does_not_overlap_and_respects_n():
    from kronosquant.evaluation import select_windows

    ts = pd.date_range("2024-01-01 09:35", periods=200, freq="5min").to_series().reset_index(drop=True)
    ts.iloc[100:] += pd.Timedelta(hours=16)  # overnight gap between rows 99 and 100
    candles = pd.DataFrame({"timestamps": ts})
    ws = select_windows(candles, range(0, 200), lookback=20, pred_len=10, n=1000)
    assert ws and all(not has_gap(candles, w.future) for w in ws)
    assert all(not (w.future.start < 100 < w.future.stop) for w in ws)  # nothing straddles the gap
    futures = [set(w.future) for w in ws]
    assert all(a.isdisjoint(b) for i, a in enumerate(futures) for b in futures[i + 1 :])
    assert len(select_windows(candles, range(0, 200), 20, 10, n=3)) == 3


def test_select_windows_bar_size_matters_for_hourly_data():
    from kronosquant.evaluation import select_windows

    candles = pd.DataFrame({"timestamps": pd.date_range("2024-01-01", periods=200, freq="1h")})
    assert select_windows(candles, range(0, 200), 20, 10, n=1000)  == []  # default 5min: every hourly step looks like a gap
    assert len(select_windows(candles, range(0, 200), 20, 10, n=1000, bar="1h")) > 10


def test_log_ratio_mae_hand_computed_and_symmetric():
    from kronosquant.metrics import log_ratio_mae

    assert log_ratio_mae(np.array([2.0, 1.0]), np.array([1.0, 2.0])) == pytest.approx(np.log(2))
    assert log_ratio_mae(np.array([3.0]), np.array([3.0])) == 0.0
    assert log_ratio_mae(np.array([2.0]), np.array([1.0])) == pytest.approx(log_ratio_mae(np.array([1.0]), np.array([2.0])))


def test_bootstrap_ic_ci_perfect_noise_and_degenerate_cases():
    from kronosquant.metrics import bootstrap_ic_ci

    x = np.random.default_rng(0).normal(size=200)
    lo, hi = bootstrap_ic_ci(x, x * 3 + 1, seed=1)
    assert lo > 0.99 and hi <= 1.0 + 1e-9
    lo, hi = bootstrap_ic_ci(x, np.random.default_rng(1).normal(size=200), level=0.99)
    assert lo < 0 < hi  # unrelated series -> interval straddles zero
    assert bootstrap_ic_ci(x, x, seed=2) == bootstrap_ic_ci(x, x, seed=2)  # seeded
    assert all(np.isnan(bootstrap_ic_ci(np.zeros(50), x[:50])))  # constant prediction -> undefined


def test_top_k_is_in_the_cache_key_only_when_used():
    from kronosquant.cache import config_key
    from kronosquant.forecast import forecast_config, load_sample_candles

    c = load_sample_candles()
    base = forecast_config(c, "cpu", 400, 12, 1.0, 0.9, 5)
    assert "top_k" not in base  # keeps keys of caches created before top_k existed
    assert config_key(forecast_config(c, "cpu", 400, 12, 1.0, 0.9, 5, top_k=1)) != config_key(base)
