"""Stage 3: forecast-quality metrics. These say nothing about trading profit."""
import numpy as np
import pandas as pd


def cumulative_log_returns(last_close: float, closes: np.ndarray) -> np.ndarray:
    """log(close_h / last_close) for each future close."""
    return np.log(np.asarray(closes, dtype=float) / last_close)


def path_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error over every window and every step. Arrays: (windows, horizon)."""
    return float(np.abs(pred - actual).mean())


def end_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean absolute error of the return at the end of the horizon only."""
    return float(np.abs(pred[:, -1] - actual[:, -1]).mean())


def directional_accuracy(pred_end: np.ndarray, actual_end: np.ndarray) -> tuple[float, int]:
    """Share of windows where the predicted sign matches the actual sign, and how many windows counted.

    Windows where either side is exactly zero are skipped (no direction to compare).
    """
    keep = (pred_end != 0) & (actual_end != 0)
    n = int(keep.sum())
    if n == 0:
        return float("nan"), 0
    return float((np.sign(pred_end[keep]) == np.sign(actual_end[keep])).mean()), n


def information_coefficient(pred_end: np.ndarray, actual_end: np.ndarray) -> float:
    """Spearman rank correlation between predicted and actual end-of-horizon returns.

    NaN when the prediction is constant (e.g. the random walk): a constant ranks nothing.
    """
    if len(pred_end) < 3 or np.ptp(pred_end) == 0:
        return float("nan")
    # Spearman = Pearson correlation of the ranks (ties get their average rank).
    ranks = pd.DataFrame({"p": pred_end, "a": actual_end}).rank()
    return float(np.corrcoef(ranks["p"], ranks["a"])[0, 1])


def paired_bootstrap_ci(diff: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the mean of per-window differences, resampling windows with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def log_ratio_mae(pred: np.ndarray, actual: np.ndarray) -> float:
    """Mean |log(pred / actual)| for positive quantities such as price range: 0.1 means 'off by about 10%'. Symmetric in over/under."""
    return float(np.abs(np.log(pred / actual)).mean())
