"""Stage 6: turn one forecast per window into positions for the Stage 5 engine.

One forecast is made after the close of bar o-1 (the last bar the model sees). The engine trades it at the open
of bar o and the position is held for `pred_len` bars, the same span the forecast covered. The exit signal is bar
o+pred_len-1 = 0, so the engine sells at the open of bar o+pred_len. Between windows the strategy is flat.
"""
import numpy as np
import pandas as pd


def directions_from_forecast(pred_end: np.ndarray, threshold: float = 0.0, allow_short: bool = False) -> np.ndarray:
    """+1 if the forecast end-of-window return is above `threshold`, -1 (or 0 if shorting is off) if below -threshold, else 0."""
    pred_end = np.asarray(pred_end, dtype=float)
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    out = np.zeros(len(pred_end))
    out[pred_end > threshold] = 1.0
    if allow_short:
        out[pred_end < -threshold] = -1.0
    return out


def window_signal(n_rows: int, origins: list[int], directions: np.ndarray, pred_len: int) -> pd.Series:
    """Signal series for `run_backtest`: direction[i] is set after bar origins[i]-1 and held for `pred_len` bars."""
    origins = list(origins)
    directions = np.asarray(directions, dtype=float)
    if len(origins) != len(directions):
        raise ValueError("one direction per window origin")
    if not np.isin(directions, (-1.0, 0.0, 1.0)).all():
        raise ValueError("directions must be -1, 0 or 1")
    if origins and origins[0] < 1:
        raise ValueError("a window needs at least one bar of history before its origin")
    if any(b - a < pred_len for a, b in zip(origins, origins[1:])):
        raise ValueError("windows must be sorted and must not overlap")
    if origins and origins[-1] + pred_len >= n_rows:
        raise ValueError("the exit bar (origin + pred_len) must exist in the data")
    signal = np.zeros(n_rows)
    for o, d in zip(origins, directions):
        signal[o - 1 : o - 1 + pred_len] = d  # decided at the close of o-1, so nothing earlier is touched
    return pd.Series(signal)


def window_returns(net_ret: np.ndarray, origins: list[int], pred_len: int) -> np.ndarray:
    """Compounded net return per window. A window owns bars origin..origin+pred_len (entry, hold and exit bars),
    stopping early if the next window starts first; flat bars between windows return 0, so these multiply to total net return."""
    out = []
    for i, o in enumerate(origins):
        end = o + pred_len
        if i + 1 < len(origins):
            end = min(end, origins[i + 1] - 1)
        out.append(float(np.prod(1.0 + net_ret[o : end + 1]) - 1.0))
    return np.array(out)
