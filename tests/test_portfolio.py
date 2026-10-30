import numpy as np
import pandas as pd
import pytest

from kronosquant.portfolio import decision_indices, long_only_weights, period_stats, run_portfolio, trailing_return, tsmom_weights, xs_weights


def panel(n=40, assets=("A", "B", "C"), seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D")
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, len(assets))), axis=0)), index=idx, columns=list(assets))
    open_ = close.shift(1).fillna(close.iloc[0])
    return open_, close


def test_trailing_return_hand_computed_and_nan_until_enough_history():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D")
    close = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1, 146.41], "B": [np.nan, np.nan, 50.0, 55.0, 60.5]}, index=idx)
    tr = trailing_return(close, 2)
    assert tr["A"].iloc[:2].isna().all() and tr["A"].iloc[2] == pytest.approx(0.21) and tr["A"].iloc[4] == pytest.approx(146.41 / 121 - 1)
    assert tr["B"].iloc[:4].isna().all() and tr["B"].iloc[4] == pytest.approx(60.5 / 50 - 1)  # B has no close two days before until day 4


def test_tsmom_and_long_only_weights():
    w = tsmom_weights(np.array([0.1, -0.2, np.nan, 0.0, 0.05]))
    np.testing.assert_allclose(w, [0.25, -0.25, 0.0, 0.0, 0.25])  # n = 4 scored assets; a zero score takes no position
    np.testing.assert_allclose(long_only_weights(np.array([0.1, -0.2, np.nan, 0.0, 0.05])), [0.25, 0.0, 0.0, 0.0, 0.25])
    assert (tsmom_weights(np.full(3, np.nan)) == 0).all()


def test_xs_weights_are_dollar_neutral_gross_one_and_pick_the_extremes():
    row = np.array([0.5, -0.4, 0.1, 0.3, -0.1, 0.0, np.nan])
    w = xs_weights(row)  # 6 scored assets, k = 2
    assert w.sum() == pytest.approx(0.0) and np.abs(w).sum() == pytest.approx(1.0)
    assert set(np.flatnonzero(w > 0)) == {0, 3} and set(np.flatnonzero(w < 0)) == {1, 4}
    assert w[6] == 0  # the NaN asset is never held
    assert (xs_weights(np.array([0.1, 0.2, 0.3, 0.4, np.nan])) == 0).all()  # fewer than 6 assets: no trade


def test_run_portfolio_hand_computed_with_entry_cost_and_turnover():
    idx = pd.date_range("2024-01-01", periods=7, freq="1D")
    open_ = pd.DataFrame({"A": [10, 10, 11, 11, 12, 12, 13], "B": [20, 20, 19, 19, 18, 18, 20]}, index=idx, dtype=float)
    scores = pd.DataFrame({"A": [1.0] * 7, "B": [-1.0] * 7}, index=idx)  # always long A, short B
    res, wts = run_portfolio(open_, scores, tsmom_weights, decisions=[0, 2], step=2, fee_bps=10)
    # decision 0: trade at open[1], exit open[3]: A 10 -> 11 (+10%), B 20 -> 19 (-5%): 0.5*0.10 - 0.5*(-0.05) = 0.075; cost 1.0 * 10 bps
    assert res["gross"].iloc[0] == pytest.approx(0.075) and res["cost"].iloc[0] == pytest.approx(0.001) and res["net"].iloc[0] == pytest.approx(0.074)
    # decision 2: trade at open[3], exit open[5]: A 11 -> 12, B 19 -> 18: same weights, so no turnover and no cost
    assert res["turnover"].iloc[1] == 0.0 and res["cost"].iloc[1] == 0.0
    assert res["gross"].iloc[1] == pytest.approx(0.5 * (12 / 11 - 1) - 0.5 * (18 / 19 - 1))
    np.testing.assert_allclose(wts.iloc[0].to_numpy(), [0.5, -0.5])


def test_weights_decided_on_day_d_cannot_see_later_prices():
    open_, close = panel(60)
    scores = trailing_return(close, 5)
    dec = decision_indices(open_.index, "2024-01-20", 5)
    _, base_w = run_portfolio(open_, scores, tsmom_weights, dec, 5, 10)
    cut = dec[3]  # a real decision day: poison everything AFTER it, so a one-day peek would read poisoned data
    factors = np.array([7.0, 0.1, 7.0])  # after the cut: asset A jumps, B collapses, C jumps (flips the momentum signs)
    open_p, close_p = open_.copy(), close.copy()
    open_p.iloc[cut + 1 :] *= factors
    close_p.iloc[cut + 1 :] *= factors
    assert not np.array_equal(np.sign(trailing_return(close_p, 5).iloc[cut + 1].to_numpy()), np.sign(scores.iloc[cut + 1].to_numpy()))  # the poison really changes the next day's signal
    _, w_p = run_portfolio(open_p, trailing_return(close_p, 5), tsmom_weights, dec, 5, 10)
    early = [d for d in dec if d <= cut]
    pd.testing.assert_frame_equal(base_w.iloc[: len(early)], w_p.iloc[: len(early)])  # decisions made at or before the cut are unchanged


def test_periods_without_an_exit_open_are_dropped_and_costs_validated():
    open_, close = panel(20)
    res, _ = run_portfolio(open_, trailing_return(close, 3), tsmom_weights, decision_indices(open_.index, "2024-01-05", 5), 5, 0)
    assert res["end"].max() <= open_.index[-1] and len(res) > 0
    with pytest.raises(ValueError):
        run_portfolio(open_, trailing_return(close, 3), tsmom_weights, [4], 5, -1)


def test_random_signals_have_no_edge_on_driftless_prices_and_costs_only_hurt():
    open_, close = panel(400, assets=tuple("ABCDEFGH"), seed=2)
    mask = trailing_return(close, 5).isna().to_numpy()
    dec = decision_indices(open_.index, "2024-02-01", 5)
    means = []
    for seed in range(60):
        rnd = pd.DataFrame(np.where(mask, np.nan, np.random.default_rng(seed).normal(size=mask.shape)), index=open_.index, columns=open_.columns)
        res, _ = run_portfolio(open_, rnd, xs_weights, dec, 5, 0)
        means.append(res["net"].mean())
    assert abs(np.mean(means)) < 0.003  # random rankings average to about zero gross
    sig = trailing_return(close, 5)
    free, _ = run_portfolio(open_, sig, xs_weights, dec, 5, 0)
    paid, _ = run_portfolio(open_, sig, xs_weights, dec, 5, 20)
    assert paid["net"].mean() < free["net"].mean() and paid["cost"].sum() > 0


def test_rank_blend_hand_computed_masks_missing_and_is_row_local():
    from kronosquant.portfolio import rank_blend

    idx = pd.date_range("2024-01-01", periods=3, freq="1D")
    a = pd.DataFrame({"X": [1.0, 5.0, 1.0], "Y": [2.0, 4.0, 2.0], "Z": [3.0, np.nan, 3.0]}, index=idx)
    b = pd.DataFrame({"X": [3.0, 1.0, 9.0], "Y": [2.0, 2.0, 8.0], "Z": [1.0, 3.0, 7.0]}, index=idx)
    out = rank_blend(a, b)
    np.testing.assert_allclose(out.iloc[0].to_numpy(), [2 / 3, 2 / 3, 2 / 3])  # opposite orderings cancel
    assert np.isnan(out.iloc[1]["Z"]) and out.iloc[1]["X"] == pytest.approx((1.0 + 0.5) / 2) and out.iloc[1]["Y"] == pytest.approx((0.5 + 1.0) / 2)  # Z lacks a signal: ranks among X, Y only
    b2 = b.copy()
    b2.iloc[0] = [9.0, 9.0, 9.0]  # changing another date must not change this date
    pd.testing.assert_frame_equal(rank_blend(a, b2).iloc[1:], out.iloc[1:])


def test_decision_indices_and_period_stats():
    idx = pd.date_range("2024-01-01", periods=12, freq="1D")
    assert decision_indices(idx, "2024-01-03", 5) == [2, 7]
    s = period_stats(np.array([0.10, -0.10, 0.10]), 73)
    assert s["mean_bps"] == pytest.approx(333.3333, rel=1e-4) and s["total_return"] == pytest.approx(1.1 * 0.9 * 1.1 - 1)
    assert s["max_drawdown"] == pytest.approx(-0.10)
    assert np.isnan(period_stats(np.zeros(5), 73)["sharpe"])  # no variation: Sharpe undefined
