import numpy as np
import pandas as pd
import pytest

from kronosquant.splits import Window
from kronosquant.volatility import candle_range, kronos_range, log_ratio_errors, range_baselines, realized_range


def make(n=200, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    half = close * rng.uniform(0.001, 0.01, n)
    return pd.DataFrame({"timestamps": pd.date_range("2024-01-01", periods=n, freq="1h"), "open": close, "close": close, "high": close + half, "low": close - half})


def test_candle_range_hand_computed():
    df = pd.DataFrame({"high": [102.0], "low": [98.0], "close": [100.0]})
    assert candle_range(df).iloc[0] == pytest.approx(0.04)


def test_realized_range_is_the_mean_over_the_future_bars():
    rng = pd.Series(np.arange(20, dtype=float))
    w = Window(origin=10, lookback=5, pred_len=4)
    assert realized_range(rng, [w])[0] == pytest.approx(np.mean([10, 11, 12, 13]))


def test_baselines_hand_computed():
    ts = pd.Series(pd.date_range("2024-01-01", periods=30, freq="1h"))
    rng = pd.Series([1.0 if h < 12 else 3.0 for h in ts.dt.hour])  # night hours range 1, afternoon hours range 3
    w = Window(origin=24, lookback=24, pred_len=6)  # history = day 1 (24 bars), future hours 0..5 -> all "1.0" hours
    b = range_baselines(rng, ts, [w], recent=4)
    assert b["hist_all"][0] == pytest.approx(2.0)  # 12 bars of 1 and 12 of 3
    assert b["hist_recent"][0] == pytest.approx(3.0)  # hours 20..23
    assert b["same_time_of_day"][0] == pytest.approx(1.0)  # hours 0..5 averaged over history are all 1


def test_baselines_cannot_see_the_origin_or_later():
    df = make()
    ws = [Window(origin=100, lookback=60, pred_len=12), Window(origin=150, lookback=60, pred_len=12)]
    base = range_baselines(candle_range(df), df["timestamps"], ws)
    poisoned = df.copy()
    poisoned.loc[100:, ["high", "low"]] = [1e6, 1.0]  # every row at or after the first origin
    after = range_baselines(candle_range(poisoned), poisoned["timestamps"], ws[:1])
    for k in base:
        assert after[k][0] == base[k][0], f"{k} changed when rows at/after the origin were poisoned"


def test_baselines_catch_an_off_by_one_leak():
    # A deliberately leaky 'recent' baseline (includes the origin bar) must differ when that bar is poisoned.
    df = make()
    w = Window(origin=100, lookback=60, pred_len=12)
    rng = candle_range(df)
    leaky = rng.iloc[100 - 48 : 100 + 1].mean()
    poisoned = df.copy()
    poisoned.loc[100, ["high", "low"]] = [1e6, 1.0]
    assert candle_range(poisoned).iloc[100 - 48 : 101].mean() != leaky
    assert range_baselines(candle_range(poisoned), poisoned["timestamps"], [w])["hist_recent"][0] == range_baselines(rng, df["timestamps"], [w])["hist_recent"][0]


def test_kronos_range_uses_forecast_high_low_close():
    w = Window(origin=10, lookback=5, pred_len=2)
    f = pd.DataFrame({"high": [101.0, 102.0], "low": [99.0, 98.0], "close": [100.0, 100.0]})
    assert kronos_range({10: f}, [w])[0] == pytest.approx(np.mean([0.02, 0.04]))


def test_log_ratio_errors_symmetric_and_zero_when_exact():
    a = np.array([1.0, 2.0])
    np.testing.assert_allclose(log_ratio_errors(a, a), 0)
    assert log_ratio_errors(np.array([2.0]), np.array([1.0]))[0] == pytest.approx(log_ratio_errors(np.array([0.5]), np.array([1.0]))[0])
