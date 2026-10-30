"""Stage 13: a small multi-asset portfolio engine for rebalanced signal strategies on daily bars.

Timeline (same convention as the single-asset engine): a decision on day d uses closes up to and including day d; it is traded at the
OPEN of day d+1 and held until the next rebalance, which trades at the open of day d+1+step. Weights are re-set at every rebalance (drift
inside a period is ignored); the cost of a rebalance is sum(|new weight - previous weight|) * bps, charged on the same period.
Shorting is assumed free of borrow and funding costs: that is optimistic and is NOT modelled.
"""
import numpy as np
import pandas as pd


def trailing_return(close: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """close[d] / close[d - lookback] - 1 for each asset. NaN until an asset has `lookback` days of history, so an asset joins a signal
    only once it has a full lookback window (new listings cannot enter early)."""
    return close / close.shift(lookback) - 1.0


def tsmom_weights(row: np.ndarray) -> np.ndarray:
    """Time-series momentum, long/short: +1/n for assets with a positive score, -1/n for a negative one, n = assets with a score."""
    valid = ~np.isnan(row)
    w = np.zeros(len(row))
    if valid.any():
        w[valid] = np.sign(row[valid]) / valid.sum()
    return w


def long_only_weights(row: np.ndarray) -> np.ndarray:
    """Time-series momentum, long only: 1/n for assets with a positive score, cash otherwise."""
    valid = ~np.isnan(row)
    w = np.zeros(len(row))
    if valid.any():
        w[valid & (row > 0)] = 1.0 / valid.sum()
    return w


def xs_weights(row: np.ndarray, min_assets: int = 6, frac: float = 1 / 3) -> np.ndarray:
    """Cross-sectional, dollar neutral: long the top `frac` by score, short the bottom `frac`, each leg 0.5 of equity. Zeros if too few assets."""
    valid = np.flatnonzero(~np.isnan(row))
    w = np.zeros(len(row))
    if len(valid) < min_assets:
        return w
    k = max(1, int(len(valid) * frac))
    order = valid[np.argsort(row[valid], kind="stable")]
    w[order[-k:]] = 0.5 / k
    w[order[:k]] = -0.5 / k
    return w


def rank_blend(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight blend of two signals: the mean of each date's cross-sectional percentile ranks. Only assets with BOTH signals present are
    scored (NaN elsewhere), and ranks are taken within that set, so the blend at a date uses nothing but that date's rows."""
    valid = a.notna() & b.notna()
    ra = a.where(valid).rank(axis=1, pct=True)
    rb = b.where(valid).rank(axis=1, pct=True)
    return (ra + rb) / 2.0


def decision_indices(dates: pd.DatetimeIndex, first_decision: str, step: int) -> list[int]:
    """Row positions of the decision days: every `step` days from the first date on or after `first_decision`."""
    start = int(dates.searchsorted(pd.Timestamp(first_decision)))
    return list(range(start, len(dates), step))


def run_portfolio(open_: pd.DataFrame, scores: pd.DataFrame, weight_fn, decisions: list[int], step: int, fee_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebalance on each decision day and return (per-period results, weights decided on each day).

    `scores` are the signal values known at each day's close (e.g. trailing returns). Rows after a day are never looked at when
    that day's weights are chosen. Periods whose exit open does not exist yet are dropped, so nothing is extrapolated.
    """
    if fee_bps < 0:
        raise ValueError("fee_bps must be >= 0")
    o, s = open_.to_numpy(float), scores.to_numpy(float)
    n = len(o)
    prev = np.zeros(o.shape[1])
    rows, weights = [], []
    for d in decisions:
        t0, t1 = d + 1, d + 1 + step
        if t1 >= n:
            break
        w = weight_fn(s[d])
        r = o[t1] / o[t0] - 1.0
        ok = ~np.isnan(r)  # an asset without prices at both ends earns nothing (its weight is not redistributed)
        gross = float((w[ok] * r[ok]).sum())
        turnover = float(np.abs(w - prev).sum())
        cost = turnover * fee_bps / 1e4
        rows.append({"decision": open_.index[d], "start": open_.index[t0], "end": open_.index[t1], "gross": gross, "cost": cost, "net": gross - cost,
                     "turnover": turnover, "n_positions": int((w != 0).sum())})
        weights.append(w)
        prev = w
    return pd.DataFrame(rows), pd.DataFrame(weights, index=[open_.index[d] for d in decisions[: len(weights)]], columns=open_.columns)


def period_stats(net: np.ndarray, periods_per_year: float) -> dict:
    """Mean, annualised Sharpe (risk-free 0), compounded return and max drawdown of per-period net returns."""
    net = np.asarray(net, dtype=float)
    sd = net.std(ddof=1) if len(net) > 1 else 0.0
    equity = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(equity)
    return {
        "mean_bps": float(net.mean() * 1e4),
        "sharpe": float(net.mean() / sd * np.sqrt(periods_per_year)) if sd > 1e-12 else float("nan"),
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": float((equity / peak - 1.0).min()),
    }
