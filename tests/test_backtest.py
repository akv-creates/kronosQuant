import numpy as np
import pandas as pd
import pytest

from kronosquant.backtest import max_drawdown, run_backtest, sharpe_ratio


def candles(opens, closes):
    n = len(opens)
    return pd.DataFrame(
        {
            "timestamps": pd.date_range("2024-01-01 09:35", periods=n, freq="5min"),
            "open": opens, "close": closes,
            "high": np.maximum(opens, closes), "low": np.minimum(opens, closes), "volume": 1.0,
        }
    )


def sig(df, values):
    return pd.Series(values, index=df.index, dtype=float)


# Four bars. Hand-checked numbers: bar1 open 100->close 110, bar2 open 110->close 99, bar3 open 99->close 99.
C = candles([100.0, 100.0, 110.0, 99.0], [100.0, 110.0, 99.0, 99.0])


def run(signal_values, fee=0.0, slip=0.0, df=C, **kw):
    return run_backtest(df, sig(df, signal_values), fee_bps=fee, slippage_bps=slip, **kw)


def test_flat_signal_never_trades_and_equity_is_constant():
    r = run([0, 0, 0, 0], fee=10, slip=5)
    assert (r.frame["equity"] == 1.0).all()
    assert r.metrics["n_rebalances"] == 0 and r.metrics["total_cost"] == 0 and r.metrics["net_return"] == 0


def test_signal_trades_at_next_open_not_at_the_close_it_was_computed_from():
    # Signal 1 after bar 0 -> buy at bar 1's open (100), earn 100->110 in bar 1, then hold.
    r = run([1, 1, 1, 1])
    np.testing.assert_allclose(r.frame["position"], [0, 1, 1, 1])
    # bar1: 0.10; bar2: carry 110/110-1=0 plus intrabar 99/110-1; bar3: carry 99/99-1=0 plus intrabar 0
    np.testing.assert_allclose(r.frame["gross_ret"], [0.0, 0.10, 99 / 110 - 1, 0.0], atol=1e-12)
    assert r.metrics["net_return"] == pytest.approx(99 / 100 - 1)  # buy at open of bar1 (100), end at close 99


def test_costs_are_charged_on_each_rebalance_and_hand_computed():
    # Buy at bar1 open, sell at bar2 open (signal 1 then 0). Two rebalances, each |change| = 1.
    r = run([1, 0, 0, 0], fee=10, slip=5)  # 15 bps per unit traded
    np.testing.assert_allclose(r.frame["cost"], [0, 0.0015, 0.0015, 0])
    # bar1: pay 15bps at the open, then earn +10% on what is left; bar2: sell at the open, pay 15bps, no move.
    np.testing.assert_allclose(r.frame["net_ret"], [0, 1.10 * 0.9985 - 1, -0.0015, 0.0], atol=1e-12)
    assert r.metrics["net_return"] == pytest.approx(1.10 * 0.9985**2 - 1)
    assert r.metrics["n_rebalances"] == 2 and r.metrics["turnover_total"] == 2.0


def test_higher_costs_never_improve_the_result():
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 500)))
    df = candles(np.r_[close[0], close[:-1]], close)
    s = (rng.random(500) > 0.5).astype(float)
    results = [run_backtest(df, sig(df, s), fee_bps=f, slippage_bps=0).metrics["net_return"] for f in (0, 1, 5, 20)]
    assert results == sorted(results, reverse=True) and results[0] > results[-1]


def test_zero_cost_buy_and_hold_equals_close_over_first_traded_open():
    rng = np.random.default_rng(1)
    open_ = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 300)))
    close = open_ * np.exp(rng.normal(0, 0.001, 300))
    df = candles(open_, close)
    r = run_backtest(df, sig(df, np.ones(300)), fee_bps=0, slippage_bps=0)
    assert r.frame["equity"].iloc[-1] == pytest.approx(close[-1] / open_[1])


def test_engine_is_causal_changing_future_signals_cannot_change_past_returns():
    rng = np.random.default_rng(2)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, 200)))
    df = candles(np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.0005, 200)), close)
    base = rng.random(200)
    j = 120
    changed = base.copy()
    changed[j:] = rng.random(200 - j)  # rewrite the "future" from index j on
    a = run_backtest(df, sig(df, base), fee_bps=5, slippage_bps=2).frame
    b = run_backtest(df, sig(df, changed), fee_bps=5, slippage_bps=2).frame
    # Bar k uses signal[k-1] and signal[k-2], so bars 0..j are unaffected by signal[j:].
    pd.testing.assert_frame_equal(a.iloc[: j + 1], b.iloc[: j + 1])
    assert not a.iloc[j + 1 :].equals(b.iloc[j + 1 :])  # and the change does matter later (test has teeth)


def test_short_position_profits_when_price_falls_and_needs_the_flag():
    df = candles([100.0, 100.0, 90.0], [100.0, 90.0, 90.0])  # open of each bar = previous close (no gaps)
    with pytest.raises(ValueError, match="no shorting"):
        run_backtest(df, sig(df, [-1, -1, -1]), fee_bps=0, slippage_bps=0)
    r = run_backtest(df, sig(df, [-1, -1, -1]), fee_bps=0, slippage_bps=0, allow_short=True)
    assert r.metrics["net_return"] == pytest.approx(0.10)


@pytest.mark.parametrize(
    "values, kwargs, match",
    [
        ([2, 0, 0, 0], {}, "no leverage"),
        ([-0.5, 0, 0, 0], {}, "no shorting"),
        ([-2, 0, 0, 0], {"allow_short": True}, "no leverage"),
        ([np.nan, 0, 0, 0], {}, "NaN"),
    ],
)
def test_signal_validation(values, kwargs, match):
    with pytest.raises(ValueError, match=match):
        run(values, **kwargs)


def test_cost_arguments_are_required_and_nonnegative():
    with pytest.raises(TypeError):
        run_backtest(C, sig(C, [0, 0, 0, 0]))  # no frictionless default
    with pytest.raises(ValueError):
        run(([0, 0, 0, 0]), fee=-1)


def test_signal_must_align_with_candles():
    with pytest.raises(ValueError, match="same index"):
        run_backtest(C, pd.Series([0.0, 0.0, 0.0]), fee_bps=0, slippage_bps=0)


def test_max_drawdown_hand_computed():
    equity = np.array([1.0, 1.2, 0.9, 1.0, 1.5, 1.2])
    assert max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1)  # peak 1.2 -> trough 0.9 = -25%
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == 0.0


def test_sharpe_hand_computed_and_degenerate_cases():
    r = np.array([0.01, -0.01, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r, 252) == pytest.approx(expected)
    assert np.isnan(sharpe_ratio(np.full(10, 0.001), 252))  # no variation -> undefined, not inf
