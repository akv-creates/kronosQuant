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


def bootstrap_ic_ci(pred_end: np.ndarray, actual_end: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the rank correlation, resampling windows (pred, actual pairs) with replacement.

    Ranks are computed once on the original data and the bootstrap resamples those ranks (Pearson on ranks), which
    avoids tie problems that duplicated resampled points would cause. NaN pair if the prediction is constant.
    """
    if len(pred_end) < 3 or np.ptp(pred_end) == 0:
        return float("nan"), float("nan")
    ranks = pd.DataFrame({"p": pred_end, "a": actual_end}).rank()
    rp, ra = ranks["p"].to_numpy(), ranks["a"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(rp), size=(n_boot, len(rp)))
    p, a = rp[idx], ra[idx]
    p, a = p - p.mean(axis=1, keepdims=True), a - a.mean(axis=1, keepdims=True)
    denom = np.sqrt((p**2).sum(axis=1) * (a**2).sum(axis=1))
    ic = np.where(denom > 0, (p * a).sum(axis=1) / np.where(denom > 0, denom, 1), np.nan)
    lo, hi = np.nanquantile(ic, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_ic_diff_ci(pred_a: np.ndarray, pred_b: np.ndarray, actual_end: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for IC(pred_a) - IC(pred_b) on the SAME windows, resampling windows with replacement.

    Both IC values use the same resampled windows each time, so the interval reflects the difference, not two separate noises.
    """
    if len(actual_end) < 3 or np.ptp(pred_a) == 0 or np.ptp(pred_b) == 0:
        return float("nan"), float("nan")
    ranks = pd.DataFrame({"a": pred_a, "b": pred_b, "y": actual_end}).rank()
    ra, rb, ry = ranks["a"].to_numpy(), ranks["b"].to_numpy(), ranks["y"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(ry), size=(n_boot, len(ry)))
    y = ry[idx] - ry[idx].mean(axis=1, keepdims=True)

    def ic(r):
        p = r[idx] - r[idx].mean(axis=1, keepdims=True)
        denom = np.sqrt((p**2).sum(axis=1) * (y**2).sum(axis=1))
        return np.where(denom > 0, (p * y).sum(axis=1) / np.where(denom > 0, denom, 1), np.nan)

    lo, hi = np.nanquantile(ic(ra) - ic(rb), [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def stationary_bootstrap_ci(x: np.ndarray, mean_block: float = 4.0, n_boot: int = 5000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Confidence interval for the mean of a time series using the stationary block bootstrap (Politis and Romano, 1994).

    Resamples blocks of consecutive observations with random (geometric) lengths averaging `mean_block`, so serial dependence and
    volatility clustering are kept inside the blocks. For independent data it is close to the ordinary bootstrap; for dependent data
    the interval is wider, which is the honest answer.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3 or mean_block < 1:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    restart = rng.random((n_boot, n)) < 1.0 / mean_block
    starts = rng.integers(0, n, size=(n_boot, n))
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
