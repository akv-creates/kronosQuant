"""Stage 5: a deliberately small, vectorized backtester for one asset.

Timeline for bar t (rows of `candles`):
  * signal[t] is what the strategy wants after seeing bar t's CLOSE.
  * It is traded at the OPEN of bar t+1 (never earlier; the engine does the shifting, not the strategy).
  * The position held during bar t is therefore signal[t-1]; the position before that is signal[t-2].
  * Bar t return = old position * (open[t]/close[t-1] - 1)   -> carried over from the previous bar
                 + new position * (close[t]/open[t] - 1)     -> earned after the rebalance at the open
                 - |new - old position| * (fee + slippage)   -> cost of that rebalance

Positions are fractions of equity in [0, 1] (long only) or [-1, 1] with allow_short. There is no leverage.
Weights are treated as re-set at every open (the usual simplification: drift within a bar is ignored).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    frame: pd.DataFrame  # per bar: position, gross_ret, cost, net_ret, equity
    metrics: dict


def max_drawdown(equity: np.ndarray) -> float:
    """Worst peak-to-trough fall as a negative fraction (e.g. -0.25 = down 25% from a prior peak)."""
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    """mean/std of per-bar returns, scaled by sqrt(bars per year). Risk-free rate taken as 0. NaN if no variation."""
    sd = returns.std(ddof=1) if len(returns) > 1 else 0.0
    if sd < 1e-12:  # constant returns carry float noise, not real variation; Sharpe is undefined there
        return float("nan")
    return float(returns.mean() / sd * np.sqrt(periods_per_year))


def infer_periods_per_year(timestamps: pd.Series) -> float:
    """Bars per calendar year implied by the data itself (overnight and weekend gaps included)."""
    years = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / (365.25 * 86400)
    return (len(timestamps) - 1) / years


def run_backtest(
    candles: pd.DataFrame,
    signal: pd.Series,
    *,
    fee_bps: float,
    slippage_bps: float,
    allow_short: bool = False,
    periods_per_year: float | None = None,
) -> BacktestResult:
    """Simulate trading `signal` on `candles`. Costs are required arguments: there is no frictionless default."""
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("costs must be >= 0")
    if len(signal) != len(candles) or not signal.index.equals(candles.index):
        raise ValueError("signal must have the same index as candles")
    target = signal.to_numpy(dtype=float)
    if np.isnan(target).any():
        raise ValueError("signal contains NaN; decide explicitly what 'no view' means (e.g. 0 = flat)")
    lo = -1.0 if allow_short else 0.0
    if target.min() < lo or target.max() > 1.0:
        raise ValueError(f"signal must be within [{lo}, 1]: no leverage" + ("" if allow_short else ", no shorting"))

    open_, close = candles["open"].to_numpy(float), candles["close"].to_numpy(float)
    pos = np.concatenate([[0.0], target[:-1]])  # held during bar t: decided at close of t-1, traded at open of t
    prev_pos = np.concatenate([[0.0], pos[:-1]])
    carry = np.zeros(len(close))
    carry[1:] = open_[1:] / close[:-1] - 1.0
    intrabar = close / open_ - 1.0

    # Exact compounding inside the bar: carry first, then pay costs at the open on the equity held then, then intrabar.
    gross = (1.0 + prev_pos * carry) * (1.0 + pos * intrabar) - 1.0
    traded = np.abs(pos - prev_pos)
    cost = traded * (fee_bps + slippage_bps) / 1e4  # fraction of equity paid at the rebalance
    net = (1.0 + gross) * (1.0 - cost) - 1.0
    equity = np.cumprod(1.0 + net)

    frame = pd.DataFrame(
        {"position": pos, "gross_ret": gross, "cost": cost, "net_ret": net, "equity": equity},
        index=candles["timestamps"].reset_index(drop=True),
    )
    ppy = periods_per_year or infer_periods_per_year(candles["timestamps"])
    years = len(net) / ppy
    active = (np.abs(pos) > 0) | (np.abs(prev_pos) > 0)
    metrics = {
        "net_return": float(equity[-1] - 1.0),
        "gross_return": float(np.prod(1.0 + gross) - 1.0),
        # annualising less than a quarter of a year is meaningless (and overflows on tiny samples)
        "ann_return": float(equity[-1] ** (1.0 / years) - 1.0) if equity[-1] > 0 and years >= 0.25 else float("nan"),
        "sharpe": sharpe_ratio(net, ppy),
        "max_drawdown": max_drawdown(equity),
        "turnover_total": float(traded.sum()),  # sum of |position change|: 1.0 = one full in-or-out trade
        "turnover_per_year": float(traded.sum() / years),
        "n_rebalances": int((traded > 0).sum()),
        "exposure": float(np.abs(pos).mean()),
        "hit_rate": float((net[active] > 0).mean()) if active.any() else float("nan"),
        "total_cost": float(cost.sum()),
        "periods_per_year": float(ppy),
    }
    return BacktestResult(frame, metrics)
