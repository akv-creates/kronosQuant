"""Stage 8: data preparation for the dashboard. No Streamlit import here, so everything is unit-testable.

The dashboard only READS result files the experiment scripts wrote (outputs/*.csv) and re-runs the cheap paper account on the
VALIDATION split. The test split is never loaded: `validation_candles` is the only way the dashboard gets price data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from kronosquant.experiments import EXPERIMENTS  # noqa: F401  (re-exported: the dashboard and tests import it from here)
from kronosquant.splits import chronological_split

LEVEL_NOTE = "99% bootstrap interval"


def validation_candles(candles: pd.DataFrame) -> pd.DataFrame:
    """Only the validation rows, re-indexed from 0. Train and test rows are not returned."""
    val = chronological_split(len(candles))["val"]
    return candles.iloc[val.start : val.stop].reset_index(drop=True)


def load_table(path: Path) -> pd.DataFrame | None:
    """A result CSV, or None when that experiment has not been run on this machine (outputs/ is git-ignored)."""
    return pd.read_csv(path) if Path(path).exists() else None


def status_of(lo: float, hi: float, lower_is_better: bool = False) -> str:
    """Three-state reading of a 99% interval against the no-skill value 0.

    Passed: the whole interval is on the good side (lower end above 0, or for lower-is-better measures the upper end below 0).
    Failed: the whole interval is on the bad side (reliably wrong). Inconclusive: the interval includes 0, so the test could not tell.
    """
    good, bad = (hi < 0, lo > 0) if lower_is_better else (lo > 0, hi < 0)
    return "Passed" if good else "Failed" if bad else "Inconclusive"


GROUP_IC, GROUP_VOL, GROUP_NET = "Information coefficient", "Volatility error difference", "Net return per period (bps)"


def scorecard(outputs: Path) -> pd.DataFrame:
    """One row per pre-declared primary comparison: what was tested, the estimate, its interval, and its three-state status.

    Built from the saved CSVs, so the dashboard cannot disagree with what the scripts produced. Missing files are skipped.
    `passed` is the declared pass/fail (interval lower end above 0, or upper end below 0 for lower-is-better measures);
    `status` adds the distinction between 'Failed' (reliably on the wrong side) and 'Inconclusive' (could not tell).
    """
    rows = []

    def add(exp_id, experiment, group, signal, estimate, lo, hi, lower_is_better=False):
        status = status_of(lo, hi, lower_is_better)
        rows.append({"exp_id": exp_id, "experiment": experiment, "group": group, "signal": signal, "estimate": float(estimate), "ci_lo": float(lo),
                     "ci_hi": float(hi), "passed": status == "Passed", "status": status, "lower_is_better": lower_is_better})

    grid = load_table(outputs / "grid_BTCUSDT_1h.csv")
    if grid is not None:
        for r in grid.itertuples():
            add("exp_grid_btc1h", "Stage 3 grid: IC on BTC 1h", GROUP_IC, r.variant, r.IC, r.IC_lo99, r.IC_hi99)
    s6 = load_table(outputs / "stage6_strategies.csv")
    if s6 is not None:
        p = s6[(s6["strategy"] == "kronos_longshort") & (s6["cost_bps"] == 10)]
        if len(p):
            r = p.iloc[0]
            add("exp_stage6_window_strategy", "Stage 6: Kronos long/short at 10 bps", GROUP_NET, "kronos_longshort", r["mean_window_bps"], r["ci_lo99_bps"], r["ci_hi99_bps"])
    s13 = load_table(outputs / "stage13_screen.csv")
    if s13 is not None:
        p13 = s13[(s13["primary"]) & (s13["window"] == "primary") & (s13["cost_bps"] == 10)]
        for r in p13.itertuples():
            add("exp_stage13_signal_screen", "Stage 13a: daily signals, net per 5-day period at 10 bps", GROUP_NET, r.candidate, r.mean_bps, r.ci_lo99_bps, r.ci_hi99_bps)
    s9 = load_table(outputs / "stage9_evaluate_val.csv")
    if s9 is not None:
        r = s9.iloc[0]
        add("exp_stage9_finetune", "Stage 9: fine-tuned IC on BTC 1h", GROUP_IC, "fine-tuned", r["IC"], r["IC_lo99"], r["IC_hi99"])
    s10 = load_table(outputs / "stage10_cross_section.csv")
    if s10 is not None:
        for r in s10.itertuples():
            add("exp_stage10_crosssection", "Stage 10: cross-asset rank IC", GROUP_IC, r.signal, r.mean_IC, r.ci_lo99, r.ci_hi99)
    s11 = load_table(outputs / "stage11_primary.csv")
    if s11 is not None:
        for r in s11.itertuples():
            add("exp_stage11_volatility", "Stage 11: volatility error vs baseline (negative = Kronos better)", GROUP_VOL, f"vs {r.baseline}", r.kronos_minus_baseline, r.ci_lo99, r.ci_hi99, True)
    s12 = load_table(outputs / "stage12_samples.csv")
    if s12 is not None and (s12["setting"] == "S20").any():
        r = s12[s12["setting"] == "S20"].iloc[0]
        add("exp_stage12_samples", "Stage 12: IC with 20 sampled paths", GROUP_IC, "S20 (20 paths)", r["IC"], r["IC_lo99"], r["IC_hi99"])
    s13b = load_table(outputs / "stage13b_incremental.csv")
    if s13b is not None:
        p = s13b[(s13b["kind"] == "paired_diff") & (s13b["primary"].astype(str) == "True")]
        if len(p):
            r = p.iloc[0]
            add("exp_stage13b_incremental", "Stage 13b: Kronos + XSMOM_28 minus XSMOM_28, net per period at 10 bps", GROUP_NET, "COMB - BASE", r["mean_bps"], r["ci_lo99_bps"], r["ci_hi99_bps"])
    return pd.DataFrame(rows, columns=["exp_id", "experiment", "group", "signal", "estimate", "ci_lo", "ci_hi", "passed", "status", "lower_is_better"])


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


def trade_markers(frame: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """Every rebalance as a row: when, at what price (the bar's OPEN, where the engine trades), and by how much.

    side is 'buy' when the position rose and 'sell' when it fell (a flip from long to short is one 'sell').
    `frame` and `candles` must be the same length and order (the paper account's frame and the candles it ran on).
    """
    if len(frame) != len(candles):
        raise ValueError("frame and candles must cover the same bars")
    pos = frame["position"].to_numpy(float)
    change = np.diff(np.concatenate([[0.0], pos]))
    idx = np.flatnonzero(np.abs(change) > 1e-12)
    return pd.DataFrame({
        "timestamp": candles["timestamps"].iloc[idx].to_numpy(),
        "price": candles["open"].iloc[idx].to_numpy(float),
        "side": np.where(change[idx] > 0, "buy", "sell"),
        "position_change": change[idx],
        "position_after": pos[idx],
    })


def buy_and_hold_equity(candles: pd.DataFrame) -> pd.Series:
    """Benchmark: enter at the first bar's open, hold, no costs. Equity starts near 1.0, like the paper account's."""
    eq = candles["close"].to_numpy(float) / float(candles["open"].iloc[0])
    return pd.Series(eq, index=candles["timestamps"].reset_index(drop=True), name="buy_and_hold")


def tripped_events(events: pd.DataFrame) -> pd.DataFrame:
    """Only the moments a limit tripped (daily loss or kill switch), for marking on the charts."""
    if events.empty:
        return events
    return events[events["rule"].isin(["daily_loss_tripped", "kill_switch_tripped"])].reset_index(drop=True)


# Placeholder strategies for the paper-account tab. They show the mechanics; they are not research findings.
def always_long(history: pd.DataFrame, position: float) -> float:
    return 1.0


def momentum_12(history: pd.DataFrame, position: float) -> float:
    return float(len(history) > 12 and history["close"].iloc[-1] > history["close"].iloc[-13])


def flat(history: pd.DataFrame, position: float) -> float:
    return 0.0


STRATEGIES = {"always long": always_long, "momentum (12 bars)": momentum_12, "flat (does nothing)": flat}


# ---- experiment table, datasets and reproducibility details ----------------------------------------------------------------------

def experiment_table(outputs: Path) -> pd.DataFrame:
    """One row per registered experiment: run state (Completed or Pending), outcome of its declared primary, date and headline number."""
    sc = scorecard(outputs)
    rows = []
    for e in EXPERIMENTS:
        files = [Path(outputs) / f for f in e["results"]]
        done = all(f.exists() for f in files)
        part = sc[sc["exp_id"] == e["id"]]
        if not done:
            outcome, headline = "Pending", ""
        elif part.empty:
            outcome, headline = "Exploratory", "no declared pass/fail rule"
        else:
            outcome = "Passed" if part["passed"].any() else "Failed" if (part["status"] == "Failed").all() else "Inconclusive"
            shown = part[part["passed"]] if part["passed"].any() else part  # a passing experiment's headline is a row that passed, never a bigger but inconclusive one
            best = shown.loc[shown["estimate"].idxmin() if shown["lower_is_better"].iloc[0] else shown["estimate"].idxmax()]
            unit = " bps" if best["group"] == GROUP_NET else ""
            headline = f"{best['signal']}: {best['estimate']:+.3f}{unit} [{best['ci_lo']:+.3f}, {best['ci_hi']:+.3f}]"
        date = max(pd.Timestamp(f.stat().st_mtime, unit="s") for f in files).strftime("%Y-%m-%d") if done else "\u2014"
        rows.append({"id": e["id"], "stage": e["stage"], "name": e["name"], "model": e["model"], "state": "Completed" if done else "Pending", "outcome": outcome,
                     "date": date, "headline": headline})
    return pd.DataFrame(rows)


def dataset_inventory(data_dir: Path) -> pd.DataFrame:
    """One row per downloaded dataset (from the .meta.json files written by fetch_binance.py plus the first and last timestamps)."""
    import json

    rows = []
    for meta_path in sorted(Path(data_dir).glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        csv = meta_path.with_name(meta_path.name.replace(".meta.json", ".csv"))
        if not csv.exists():
            continue
        ts = pd.read_csv(csv, usecols=["timestamps"], parse_dates=["timestamps"])["timestamps"]
        rows.append({"symbol": meta["symbol"], "interval": meta["interval"], "rows": meta["rows"], "first": ts.iloc[0].strftime("%Y-%m-%d"),
                     "last": ts.iloc[-1].strftime("%Y-%m-%d"), "gaps": meta["gaps"], "fingerprint": meta["fingerprint"][:12], "fetched_utc": meta["fetched_utc"][:10]})
    return pd.DataFrame(rows, columns=["symbol", "interval", "rows", "first", "last", "gaps", "fingerprint", "fetched_utc"])


def code_version(root: Path) -> str:
    """Short git commit of the checkout, marked when the working tree has uncommitted changes. 'unknown' outside a git repository."""
    import subprocess

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10)

    try:
        head = git("rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            return "unknown"
        dirty = bool(git("status", "--porcelain").stdout.strip())
        return head.stdout.strip() + (" (+ uncommitted changes)" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def declared_rule(script: Path) -> str:
    """The rule an experiment script declared before running: its module docstring from the 'DECLARED BEFORE ...' line on
    (scripts say RUNNING, EVALUATING or TRAINING). If there is no such line, the whole docstring is returned."""
    import ast

    doc = ast.get_docstring(ast.parse(Path(script).read_text())) or ""
    i = doc.find("DECLARED BEFORE")
    return doc[i:] if i >= 0 else doc


def pinned_revisions(root: Path) -> dict[str, str]:
    """The model revisions the forecasts were made with, read from forecast.py and vendor/KRONOS_COMMIT without importing torch."""
    import re

    text = (Path(root) / "src" / "kronosquant" / "forecast.py").read_text()
    out = {}
    for key, name in [("tokenizer", "TOKENIZER_REVISION"), ("model", "MODEL_REVISION")]:
        m = re.search(rf'{name}\s*=\s*"([0-9a-f]{{40}})"', text)
        out[key] = m.group(1) if m else "unknown"
    commit = Path(root) / "vendor" / "KRONOS_COMMIT"
    out["kronos_commit"] = commit.read_text().strip() if commit.exists() else "unknown"
    return out


def key_metrics(frame: pd.DataFrame) -> dict:
    """Plain-language trading statistics from a paper-account frame: turnover, cost in bps of equity, entries, average hold, hit rate."""
    pos = frame["position"].to_numpy(float)
    prev = np.concatenate([[0.0], pos[:-1]])
    exposed = np.abs(pos) > 1e-12
    entries = int(((np.abs(prev) <= 1e-12) & exposed).sum())
    net = frame["net_ret"].to_numpy(float)
    return {
        "turnover": float(np.abs(pos - prev).sum()),
        "cost_bps": float(frame["cost"].sum() * 1e4),
        "entries": entries,
        "avg_hold_bars": float(exposed.sum() / entries) if entries else float("nan"),
        "hit_rate": float((net[exposed] > 0).mean()) if exposed.any() else float("nan"),
    }
