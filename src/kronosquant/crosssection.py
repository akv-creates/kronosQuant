"""Stage 10: cross-sectional evaluation. At each date, do the forecasts RANK the assets in the order their returns turn out?

This asks a different question from the single-asset test: market-wide moves cancel out, because only the order
between assets matters on a given date.
"""
import numpy as np
import pandas as pd


def rank_ic_by_date(pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Spearman rank correlation across assets for each date. Arrays: (dates, assets). NaN for a date with no variation."""
    pred, actual = np.asarray(pred, dtype=float), np.asarray(actual, dtype=float)
    if pred.shape != actual.shape or pred.ndim != 2 or pred.shape[1] < 3:
        raise ValueError("need matching (dates, assets) arrays with at least 3 assets")
    rp = pd.DataFrame(pred.T).rank().to_numpy().T  # ties get their average rank
    ra = pd.DataFrame(actual.T).rank().to_numpy().T
    rp, ra = rp - rp.mean(axis=1, keepdims=True), ra - ra.mean(axis=1, keepdims=True)
    denom = np.sqrt((rp**2).sum(axis=1) * (ra**2).sum(axis=1))
    return np.where(denom > 0, (rp * ra).sum(axis=1) / np.where(denom > 0, denom, 1), np.nan)
