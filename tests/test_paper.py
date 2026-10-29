import numpy as np
import pandas as pd
import pytest

from kronosquant.backtest import run_backtest
from kronosquant.paper import RiskLimits, run_paper

LOOSE = RiskLimits(max_position=1.0, max_order=2.0, max_daily_loss=0.99, max_drawdown=0.99)  # nothing should bind


def make(closes, opens=None, freq="1h", start="2024-01-01"):
    closes = np.asarray(closes, float)
    opens = np.asarray(opens if opens is not None else np.concatenate([[closes[0]], closes[:-1]]), float)
    return pd.DataFrame({
        "timestamps": pd.date_range(start, periods=len(closes), freq=freq),
        "open": opens, "close": closes, "high": np.maximum(opens, closes), "low": np.minimum(opens, closes), "volume": 1.0,
    })


def scripted(values):
    """A strategy that replays a fixed wish per bar (it finds the bar from the history's last index label)."""
    return lambda history, position: values[history.index[-1]]


def random_walk(n=300, seed=0):
    rng = np.random.default_rng(seed)
    closes = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    opens = closes * np.exp(rng.normal(0, 0.002, n))  # open != previous close, so the carry term matters
    return make(closes, opens)


def test_matches_the_backtester_exactly_when_no_limit_binds():
    df = random_walk()
    wishes = np.random.default_rng(1).choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=len(df))
    paper = run_paper(df, scripted(wishes), limits=LOOSE, fee_bps=7, slippage_bps=3, allow_short=True, lookback=len(df))
    ref = run_backtest(df, pd.Series(wishes, index=df.index), fee_bps=7, slippage_bps=3, allow_short=True)
    np.testing.assert_allclose(paper.frame["position"], ref.frame["position"], atol=1e-12)
    np.testing.assert_allclose(paper.frame["equity"], ref.frame["equity"], rtol=1e-12)
    assert paper.metrics["net_return"] == pytest.approx(ref.metrics["net_return"], rel=1e-12)
    assert paper.metrics["total_cost"] == pytest.approx(ref.metrics["total_cost"], rel=1e-9)
    assert len(paper.events) == 0


def test_strategy_sees_only_the_past_and_cannot_alter_the_data():
    df = random_walk(60)
    original = df.copy()
    seen = []

    def spy(history, position):
        seen.append((history.index[-1], len(history)))
        history["close"] = -1.0  # vandalism must not reach the account's data
        return 0.0

    run_paper(df, spy, limits=LOOSE, fee_bps=0, slippage_bps=0, lookback=10)
    assert [s[0] for s in seen] == list(range(len(df) - 1))  # one call per decision, never the last bar's future
    assert max(s[1] for s in seen) == 10  # lookback respected
    pd.testing.assert_frame_equal(df, original)


def test_future_data_cannot_change_a_decision():
    df = random_walk(80)
    poisoned = df.copy()
    poisoned.loc[41:, ["open", "high", "low", "close"]] = 1e9  # everything after bar 40's close
    momentum = lambda h, p: float(len(h) >= 5 and h["close"].iloc[-1] > h["close"].iloc[-5])
    a = run_paper(df, momentum, limits=LOOSE, fee_bps=0, slippage_bps=0)
    b = run_paper(poisoned, momentum, limits=LOOSE, fee_bps=0, slippage_bps=0)
    # the position traded at bar t+1 is decided at bar t's close, so positions up to bar 41 cannot see the poison
    np.testing.assert_array_equal(a.frame["position"].to_numpy()[:42], b.frame["position"].to_numpy()[:42])


def test_oversized_and_short_wishes_are_clipped_and_logged():
    df = make([100.0] * 6)
    limits = RiskLimits(max_position=0.5, max_order=2.0, max_daily_loss=0.5, max_drawdown=0.5)
    r = run_paper(df, lambda h, p: 5.0, limits=limits, fee_bps=0, slippage_bps=0)
    assert r.frame["position"].max() == 0.5
    assert (r.events["rule"] == "max_position").all() and len(r.events) == 5
    short = run_paper(df, lambda h, p: -1.0, limits=limits, fee_bps=0, slippage_bps=0, allow_short=False)
    assert (short.frame["position"] == 0).all()  # long-only account: a short wish becomes flat
    both = run_paper(df, lambda h, p: -1.0, limits=limits, fee_bps=0, slippage_bps=0, allow_short=True)
    assert both.frame["position"].min() == -0.5


def test_order_size_limit_steps_toward_the_wish():
    df = make([100.0] * 8)
    limits = RiskLimits(max_position=1.0, max_order=0.25, max_daily_loss=0.5, max_drawdown=0.5)
    r = run_paper(df, lambda h, p: 1.0, limits=limits, fee_bps=0, slippage_bps=0)
    np.testing.assert_allclose(r.frame["position"], [0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0])


def test_non_finite_wish_means_flat_and_is_logged():
    df = make([100.0] * 5)
    wishes = {0: 1.0, 1: np.nan, 2: np.inf, 3: 1.0}
    r = run_paper(df, scripted(wishes), limits=LOOSE, fee_bps=0, slippage_bps=0)
    np.testing.assert_allclose(r.frame["position"], [0, 1, 0, 0, 1])
    assert list(r.events["rule"]) == ["invalid_wish", "invalid_wish"]


def day_candles(day_closes):
    """24 hourly bars per day; each day follows the given list of closes (len 24), continuing from the last close."""
    closes = np.concatenate(day_closes)
    return make(closes)


def test_daily_loss_halts_for_the_rest_of_the_day_and_resumes_next_day():
    day1 = [100.0] * 5 + [97.0] * 19  # a 3% drop at bar 5 against a 2% daily limit
    day2 = [97.0] * 24
    df = day_candles([day1, day2])
    limits = RiskLimits(max_position=1.0, max_order=2.0, max_daily_loss=0.02, max_drawdown=0.5)
    r = run_paper(df, lambda h, p: 1.0, limits=limits, fee_bps=0, slippage_bps=0)
    pos = r.frame["position"].to_numpy()
    assert pos[5] == 1.0 and pos[6:24].tolist() == [0.0] * 18  # the breach is seen at bar 5's close: flat from bar 6 to the end of day 1
    assert pos[24] == 1.0  # a new day has a new budget: the strategy is asked again
    assert "daily_loss_tripped" in set(r.events["rule"]) and not r.metrics["kill_switch"]
    assert r.metrics["bars_in_daily_halt"] == 18  # flagged at bars 5..22; the flag clears at bar 23's close, when the next day starts


def test_kill_switch_is_permanent_even_if_the_market_recovers():
    closes = [100.0] * 4 + [80.0] * 4 + [200.0] * 40  # -20% against a 10% drawdown limit, then a huge recovery
    df = make(closes)
    limits = RiskLimits(max_position=1.0, max_order=2.0, max_daily_loss=0.99, max_drawdown=0.10)
    r = run_paper(df, lambda h, p: 1.0, limits=limits, fee_bps=0, slippage_bps=0)
    assert r.metrics["kill_switch"] and r.frame["killed"].iloc[-1]
    first_flat = int(np.argmax((r.frame["position"].to_numpy() == 0) & (np.arange(len(df)) > 3)))
    assert (r.frame["position"].to_numpy()[first_flat:] == 0).all()  # never re-enters, whatever the strategy wishes
    assert "kill_switch_tripped" in set(r.events["rule"])


def test_risk_reducing_exit_is_never_blocked_by_the_order_size_limit():
    closes = [100.0] * 6 + [70.0] * 4 + [70.0] * 6
    df = make(closes)
    limits = RiskLimits(max_position=1.0, max_order=0.25, max_daily_loss=0.99, max_drawdown=0.10)
    r = run_paper(df, lambda h, p: 1.0, limits=limits, fee_bps=0, slippage_bps=0)
    pos = r.frame["position"].to_numpy()
    assert pos[5] == 1.0 and pos[6] > 0  # it had built the full position slowly (4 bars at 0.25)
    breach = int(np.argmax(r.frame["equity"].to_numpy() / np.maximum.accumulate(r.frame["equity"].to_numpy()) - 1 <= -0.10))
    assert pos[breach + 1] == 0.0  # flattened in ONE bar, although max_order is 0.25


def test_flattening_pays_costs_like_any_other_trade():
    closes = [100.0] * 4 + [90.0] * 10
    df = make(closes)
    limits = RiskLimits(max_position=1.0, max_order=2.0, max_daily_loss=0.99, max_drawdown=0.05)
    free = run_paper(df, lambda h, p: 1.0, limits=limits, fee_bps=0, slippage_bps=0)
    paid = run_paper(df, lambda h, p: 1.0, limits=limits, fee_bps=20, slippage_bps=0)
    assert paid.metrics["total_cost"] == pytest.approx(2 * 20 / 1e4, rel=1e-6)  # one entry and one forced exit, each 20 bps
    assert paid.metrics["net_return"] < free.metrics["net_return"]


def test_limits_and_costs_are_validated():
    for bad in [dict(max_position=1.5), dict(max_position=0), dict(max_order=3), dict(max_daily_loss=1.0), dict(max_drawdown=0)]:
        with pytest.raises(ValueError):
            RiskLimits(**{**dict(max_position=1.0, max_order=1.0, max_daily_loss=0.1, max_drawdown=0.2), **bad})
    with pytest.raises(ValueError):
        run_paper(make([100.0] * 4), lambda h, p: 0.0, limits=LOOSE, fee_bps=-1, slippage_bps=0)
