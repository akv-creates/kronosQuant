import numpy as np
import pandas as pd
import pytest

from kronosquant.backtest import run_backtest
from kronosquant.strategy import directions_from_forecast, window_returns, window_signal


def random_candles(n=300, seed=0):
    close = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]])  # open = previous close: no gaps between bars
    return pd.DataFrame(
        {"timestamps": pd.date_range("2024-01-01", periods=n, freq="1h"), "open": open_, "close": close,
         "high": np.maximum(open_, close), "low": np.minimum(open_, close), "volume": 1.0}
    )


def test_directions_threshold_and_shorting():
    pred = np.array([0.02, 0.001, -0.001, -0.02, 0.0])
    np.testing.assert_array_equal(directions_from_forecast(pred, 0.005), [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(directions_from_forecast(pred, 0.005, allow_short=True), [1, 0, 0, -1, 0])
    np.testing.assert_array_equal(directions_from_forecast(pred, 0.0, allow_short=True), [1, 1, -1, -1, 0])
    with pytest.raises(ValueError):
        directions_from_forecast(pred, -1)


def test_position_is_held_exactly_from_origin_open_for_pred_len_bars():
    df = random_candles(40)
    sig = window_signal(len(df), [5, 20], np.array([1.0, -1.0]), pred_len=3)
    assert sig.iloc[:4].eq(0).all()  # nothing is decided before the close of bar origin-1 = 4
    r = run_backtest(df, sig, fee_bps=0, slippage_bps=0, allow_short=True)
    pos = r.frame["position"].to_numpy()
    expected = np.zeros(40)
    expected[5:8] = 1.0   # bars 5,6,7 held; sold at the open of bar 8
    expected[20:23] = -1.0
    np.testing.assert_array_equal(pos, expected)


def test_window_signal_rejects_bad_inputs():
    d = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="overlap"):
        window_signal(100, [10, 12], d, 3)
    with pytest.raises(ValueError, match="history"):
        window_signal(100, [0, 12], d, 3)
    with pytest.raises(ValueError, match="exit bar"):
        window_signal(100, [10, 97], d, 3)
    with pytest.raises(ValueError, match="-1, 0 or 1"):
        window_signal(100, [10, 20], np.array([0.5, 1.0]), 3)
    with pytest.raises(ValueError, match="one direction"):
        window_signal(100, [10, 20], np.array([1.0]), 3)


@pytest.mark.parametrize("origins", [[10, 25, 60], [10, 13, 16, 40]])  # gaps, and back-to-back windows
def test_window_returns_multiply_to_total_net_return(origins):
    df = random_candles(100, seed=3)
    dirs = np.array([1.0, -1.0, 1.0, -1.0][: len(origins)])
    r = run_backtest(df, window_signal(len(df), origins, dirs, 3), fee_bps=10, slippage_bps=0, allow_short=True)
    per_window = window_returns(r.frame["net_ret"].to_numpy(), origins, 3)
    assert np.prod(1 + per_window) - 1 == pytest.approx(r.metrics["net_return"], rel=1e-9)


def test_perfect_foresight_wins_reverse_loses_and_costs_hurt():
    df = random_candles(400, seed=5)
    origins = list(range(10, 380, 12))
    close = df["close"].to_numpy()
    future = np.array([close[o + 11] / close[o - 1] - 1 for o in origins])
    right, wrong = np.sign(future), -np.sign(future)

    def net(dirs, fee):
        return run_backtest(df, window_signal(len(df), origins, dirs, 12), fee_bps=fee, slippage_bps=0, allow_short=True).metrics["net_return"]

    assert net(right, 0) > 0 > net(wrong, 0)
    assert net(right, 20) < net(right, 0)
