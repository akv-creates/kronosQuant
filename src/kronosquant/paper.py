"""Stage 7: a bar-by-bar PAPER account with risk limits the strategy cannot bypass. Simulated fills only: no broker, no keys, no network.

Per bar t, in this order (same timeline and arithmetic as backtest.run_backtest):
  1. The position decided at the close of bar t-1 is traded at the OPEN of bar t, paying (fee + slippage) on the change.
  2. Bar t's return is booked (carry with the old position, then open->close with the new one).
  3. At the CLOSE of bar t the strategy is shown history up to and including bar t and its current position, and returns a WISH.
  4. The RiskManager turns the wish into the position that will actually be traded at the open of bar t+1.

The strategy is only a function; it holds no reference to the account or the limits, and it gets a copy of the history.
Risk-reducing exits (halts, kill switch) are never blocked by the order-size limit: it is always possible to get flat.
Gap risk: a halt flattens at the NEXT open, so a loss can overshoot the limit (price can jump between close and open).
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from kronosquant.backtest import infer_periods_per_year, max_drawdown, sharpe_ratio

Strategy = Callable[[pd.DataFrame, float], float]  # (history up to this close, current position) -> wished position in [-1, 1]


@dataclass(frozen=True)
class RiskLimits:
    """All fields are required: there is no 'unlimited' default, so a loose limit is a visible decision."""

    max_position: float  # largest |position| as a fraction of equity (1.0 = fully invested; no leverage above that)
    max_order: float  # largest |change in position| per rebalance (2.0 allows a full long-to-short flip)
    max_daily_loss: float  # halt for the rest of the day when equity is this far below the day's starting equity
    max_drawdown: float  # permanent kill switch when equity is this far below its peak

    def __post_init__(self):
        if not 0 < self.max_position <= 1:
            raise ValueError("max_position must be in (0, 1]: no leverage")
        if not 0 < self.max_order <= 2:
            raise ValueError("max_order must be in (0, 2]")
        if not 0 < self.max_daily_loss < 1 or not 0 < self.max_drawdown < 1:
            raise ValueError("max_daily_loss and max_drawdown must be fractions in (0, 1)")


@dataclass(frozen=True)
class PaperResult:
    frame: pd.DataFrame  # per bar: position, cost, net_ret, equity, wish, allowed, day_halted, killed
    events: pd.DataFrame  # one row per time a rule changed or blocked something: bar, timestamp, rule, wish, allowed
    metrics: dict


def _decide(wish, position, limits, allow_short, day_halted, killed):
    """Pure risk logic: returns (allowed position, rule name or None)."""
    if killed:
        return 0.0, "kill_switch"
    if day_halted:
        return 0.0, "daily_loss_halt"
    if not np.isfinite(wish):
        return 0.0, "invalid_wish"
    allowed = float(np.clip(wish, -limits.max_position if allow_short else 0.0, limits.max_position))
    rule = "max_position" if allowed != wish else None
    step = allowed - position
    if abs(step) > limits.max_order:
        allowed = position + np.sign(step) * limits.max_order
        rule = "max_order"
    return allowed, rule


def run_paper(
    candles: pd.DataFrame,
    strategy: Strategy,
    *,
    limits: RiskLimits,
    fee_bps: float,
    slippage_bps: float,
    allow_short: bool = False,
    lookback: int = 512,
    periods_per_year: float | None = None,
) -> PaperResult:
    """Run `strategy` through a paper account. Equity starts at 1.0 and is a fraction, not a currency amount."""
    if fee_bps < 0 or slippage_bps < 0:
        raise ValueError("costs must be >= 0")
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    n = len(candles)
    open_, close = candles["open"].to_numpy(float), candles["close"].to_numpy(float)
    dates = candles["timestamps"].dt.normalize().to_numpy()
    cost_rate = (fee_bps + slippage_bps) / 1e4

    pos, target = 0.0, 0.0  # position held during the current bar; position decided at the previous close
    equity = peak = day_start = 1.0
    day_halted = killed = False
    rows, events = [], []
    for t in range(n):
        new = target
        carry = open_[t] / close[t - 1] - 1.0 if t > 0 else 0.0
        gross = (1.0 + pos * carry) * (1.0 + new * (close[t] / open_[t] - 1.0)) - 1.0
        cost = abs(new - pos) * cost_rate
        net = (1.0 + gross) * (1.0 - cost) - 1.0
        equity *= 1.0 + net
        peak = max(peak, equity)
        pos = new

        wish = allowed = np.nan
        if t < n - 1:  # the last close has no next open to trade at
            if dates[t + 1] != dates[t]:  # the trade below happens in a new day: fresh daily budget
                day_start, day_halted = equity, False
            if not killed and equity / peak - 1.0 <= -limits.max_drawdown:
                killed = True
                events.append((t, "kill_switch_tripped", np.nan, 0.0))
            if not killed and not day_halted and equity / day_start - 1.0 <= -limits.max_daily_loss:
                day_halted = True
                events.append((t, "daily_loss_tripped", np.nan, 0.0))
            asked = not (killed or day_halted)  # when halted the strategy is not even asked
            if asked:
                history = candles.iloc[max(0, t + 1 - lookback) : t + 1].copy()
                wish = float(strategy(history, pos))
            allowed, rule = _decide(wish, pos, limits, allow_short, day_halted, killed)
            if rule is not None and (asked or allowed != pos):  # a halt while already flat changes nothing: no event
                events.append((t, rule, wish, allowed))
            target = allowed
        rows.append((pos, cost, net, equity, wish, allowed, day_halted, killed))

    frame = pd.DataFrame(
        rows, columns=["position", "cost", "net_ret", "equity", "wish", "allowed", "day_halted", "killed"],
        index=candles["timestamps"].reset_index(drop=True),
    )
    ev = pd.DataFrame(events, columns=["bar", "rule", "wish", "allowed"])
    ev.insert(1, "timestamp", frame.index[ev["bar"].to_numpy(int)] if len(ev) else pd.DatetimeIndex([]))
    net_arr, eq = frame["net_ret"].to_numpy(), frame["equity"].to_numpy()
    ppy = periods_per_year or infer_periods_per_year(candles["timestamps"])
    traded = np.abs(np.diff(np.concatenate([[0.0], frame["position"].to_numpy()])))
    metrics = {
        "net_return": float(eq[-1] - 1.0),
        "max_drawdown": max_drawdown(eq),
        "sharpe": sharpe_ratio(net_arr, ppy),
        "turnover_total": float(traded.sum()),
        "total_cost": float(frame["cost"].sum()),
        "exposure": float(np.abs(frame["position"]).mean()),
        "kill_switch": bool(killed),
        "bars_in_daily_halt": int(frame["day_halted"].sum()),
        "n_events": int(len(ev)),
        "periods_per_year": float(ppy),
    }
    return PaperResult(frame, ev, metrics)
