"""Stage 3: build evaluation windows and the forecast / actual return paths to compare."""
import numpy as np
import pandas as pd

from kronosquant.data import has_gap
from kronosquant.metrics import cumulative_log_returns
from kronosquant.splits import Window, make_windows


def select_windows(candles: pd.DataFrame, split: range, lookback: int, pred_len: int, n: int, bar: str = "5min") -> list[Window]:
    """Up to `n` evenly spaced, non-overlapping windows whose forecast period has no time gap.

    Future periods are back to back (step = pred_len) so no two windows share a target candle.
    Windows that straddle lunch or overnight are dropped: those returns mix in a different process.
    `bar` is the candle size (pandas offset string, e.g. "5min", "1h"): any longer step counts as a gap.
    """
    windows = [w for w in make_windows(split, lookback, pred_len, step=pred_len) if not has_gap(candles, w.future, bar)]
    if len(windows) <= n:
        return windows
    return [windows[i] for i in np.linspace(0, len(windows) - 1, n).round().astype(int)]


def last_close(candles: pd.DataFrame, w: Window) -> float:
    return float(candles["close"].iloc[w.origin - 1])


def actual_paths(candles: pd.DataFrame, windows: list[Window]) -> np.ndarray:
    """(windows, horizon) cumulative log-returns that really happened."""
    return np.stack([cumulative_log_returns(last_close(candles, w), candles["close"].iloc[w.future].to_numpy()) for w in windows])


def baseline_paths(candles: pd.DataFrame, windows: list[Window], baseline) -> np.ndarray:
    return np.stack([baseline(candles["close"].iloc[w.history].to_numpy(), w.pred_len) for w in windows])


def paths_from_forecasts(candles: pd.DataFrame, windows: list[Window], forecasts: dict[int, pd.DataFrame]) -> np.ndarray:
    """Turn forecast candles (keyed by window origin) into cumulative log-return paths."""
    return np.stack([cumulative_log_returns(last_close(candles, w), forecasts[w.origin]["close"].to_numpy()) for w in windows])
