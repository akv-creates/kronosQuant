"""Stage 11: volatility (candle range) forecasts and their baselines. Everything here uses only rows before the window origin,
except `realized_range`, which is the answer being predicted.

Range of a candle = (high - low) / close. A window's range is the mean over its horizon bars.
"""
import numpy as np
import pandas as pd

from kronosquant.splits import Window


def candle_range(candles: pd.DataFrame) -> pd.Series:
    return (candles["high"] - candles["low"]) / candles["close"]


def realized_range(rng: pd.Series, windows: list[Window]) -> np.ndarray:
    """The target: mean range over each window's future bars."""
    return np.array([rng.iloc[w.future].mean() for w in windows])


def kronos_range(forecasts: dict[int, pd.DataFrame], windows: list[Window]) -> np.ndarray:
    """Mean forecast range over the horizon (the cache holds the mean of the sample paths, so this is an expected range)."""
    return np.array([((forecasts[w.origin]["high"] - forecasts[w.origin]["low"]) / forecasts[w.origin]["close"]).mean() for w in windows])


def range_baselines(rng: pd.Series, timestamps: pd.Series, windows: list[Window], recent: int = 48) -> dict[str, np.ndarray]:
    """Three simple forecasts of a window's mean range, built from rows before the origin only.

    hist_all:         mean range over the whole lookback history.
    hist_recent:      mean range over the last `recent` bars before the origin.
    same_time_of_day: for each future bar, the mean range of history bars at the same hour of day, averaged over the horizon.
    """
    hour = timestamps.dt.hour
    out = {"hist_all": [], "hist_recent": [], "same_time_of_day": []}
    for w in windows:
        hist = rng.iloc[w.history]
        out["hist_all"].append(hist.mean())
        out["hist_recent"].append(rng.iloc[w.origin - recent : w.origin].mean())
        by_hour = hist.groupby(hour.iloc[w.history]).mean()
        out["same_time_of_day"].append(np.mean([by_hour.get(hour.iloc[i], hist.mean()) for i in w.future]))
    return {k: np.array(v) for k, v in out.items()}


def log_ratio_errors(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """|log(pred/actual)| per window: 0.2 means off by about 20%. Symmetric between over- and under-prediction."""
    return np.abs(np.log(pred / actual))
