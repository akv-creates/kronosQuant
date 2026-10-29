"""Stage 8: data preparation for the dashboard. No Streamlit import here, so everything is unit-testable.

The dashboard only READS result files the experiment scripts wrote (outputs/*.csv) and re-runs the cheap paper account on the
VALIDATION split. The test split is never loaded: `validation_candles` is the only way the dashboard gets price data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from kronosquant.splits import chronological_split

LEVEL_NOTE = "99% bootstrap interval"


def validation_candles(candles: pd.DataFrame) -> pd.DataFrame:
    """Only the validation rows, re-indexed from 0. Train and test rows are not returned."""
    val = chronological_split(len(candles))["val"]
    return candles.iloc[val.start : val.stop].reset_index(drop=True)


def load_table(path: Path) -> pd.DataFrame | None:
    """A result CSV, or None when that experiment has not been run on this machine (outputs/ is git-ignored)."""
    return pd.read_csv(path) if Path(path).exists() else None


def scorecard(outputs: Path) -> pd.DataFrame:
    """One row per pre-declared primary comparison: what was tested, the estimate, its interval, and whether it passed.

    Built from the saved CSVs, so the dashboard cannot disagree with what the scripts produced. Missing files are skipped.
    """
    rows = []

    def add(experiment, signal, estimate, lo, hi, passed):
        rows.append({"experiment": experiment, "signal": signal, "estimate": float(estimate), "ci_lo": float(lo), "ci_hi": float(hi), "passed": bool(passed)})

    grid = load_table(outputs / "grid_BTCUSDT_1h.csv")
    if grid is not None:
        for r in grid.itertuples():
            add("Stage 3 grid: IC on BTC 1h", r.variant, r.IC, r.IC_lo99, r.IC_hi99, r.PASS_primary)
    s9 = load_table(outputs / "stage9_evaluate_val.csv")
    if s9 is not None:
        r = s9.iloc[0]
        add("Stage 9: fine-tuned IC on BTC 1h", "fine-tuned", r["IC"], r["IC_lo99"], r["IC_hi99"], r["PASS_primary"])
    s10 = load_table(outputs / "stage10_cross_section.csv")
    if s10 is not None:
        for r in s10.itertuples():
            add("Stage 10: cross-asset rank IC", r.signal, r.mean_IC, r.ci_lo99, r.ci_hi99, r.PASS_primary)
    s11 = load_table(outputs / "stage11_primary.csv")
    if s11 is not None:
        for r in s11.itertuples():
            add("Stage 11: volatility error vs baseline (negative = Kronos better)", f"vs {r.baseline}", r.kronos_minus_baseline, r.ci_lo99, r.ci_hi99, r.PASS)
    return pd.DataFrame(rows, columns=["experiment", "signal", "estimate", "ci_lo", "ci_hi", "passed"])


def equity_and_drawdown(frame: pd.DataFrame) -> pd.DataFrame:
    """Equity and drawdown (negative fraction below the running peak) per bar, from a paper-account or backtest frame."""
    equity = frame["equity"].astype(float)
    return pd.DataFrame({"equity": equity, "drawdown": equity / equity.cummax() - 1.0})


def events_by_rule(events: pd.DataFrame) -> pd.DataFrame:
    """How often each risk rule changed or blocked something, most frequent first."""
    if events.empty:
        return pd.DataFrame({"rule": [], "count": []})
    counts = events["rule"].value_counts()
    return pd.DataFrame({"rule": counts.index, "count": counts.to_numpy()})


# Placeholder strategies for the paper-account tab. They show the mechanics; they are not research findings.
def always_long(history: pd.DataFrame, position: float) -> float:
    return 1.0


def momentum_12(history: pd.DataFrame, position: float) -> float:
    return float(len(history) > 12 and history["close"].iloc[-1] > history["close"].iloc[-13])


def flat(history: pd.DataFrame, position: float) -> float:
    return 0.0


STRATEGIES = {"always long": always_long, "momentum (12 bars)": momentum_12, "flat (does nothing)": flat}
