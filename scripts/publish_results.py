"""Build `published/`: the small, committed snapshot of results that the hosted dashboard reads.

Why: `outputs/` and `data/` are git-ignored (generated results, and Binance candles whose redistribution terms are restrictive), so a fresh
deployment would otherwise show empty pages. This script copies the result CSVs unchanged, adds derived summaries that contain NO raw prices
(dataset inventory, split dates, precomputed paper-account runs) and writes `manifest.json` with a SHA-256 per file, the date each result was
generated and the commit it came from. Visitors of the hosted dashboard therefore never trigger model inference or backtests.

Run after any experiment finishes, then commit `published/`:   .venv/bin/python -I scripts/publish_results.py
Needs the local `outputs/` results and `data/` candles. It reads results only; it never changes them.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from kronosquant.backtest import sharpe_ratio  # noqa: E402
from kronosquant.config import ROOT  # noqa: E402
from kronosquant.dashboard_data import STRATEGIES, buy_and_hold_equity, dataset_inventory, equity_and_drawdown, split_table, validation_candles  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.experiments import EXPERIMENTS  # noqa: E402
from kronosquant.paper import RiskLimits, run_paper  # noqa: E402

OUT, DATA, PUBLISHED = ROOT / "outputs", ROOT / "data", ROOT / "published"
EXTRA_RESULTS = ["equity_curves.csv", "kronos_window_returns.csv"]  # written by scripts/export_equity_curves.py, used by the Home page
LIMITS = dict(max_position=0.5, max_order=0.25, max_daily_loss=0.03, max_drawdown=0.15)  # the Strategies page defaults
LIMITS_OFF = dict(max_position=1.0, max_order=2.0, max_daily_loss=0.99, max_drawdown=0.99)
COSTS_BPS = [0, 5, 10, 20, 50]
METRIC_KEYS = ["net_return", "sharpe", "max_drawdown", "exposure", "turnover_total", "total_cost", "kill_switch", "bars_in_daily_halt", "n_events"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def paper_examples(candles: pd.DataFrame) -> pd.DataFrame:
    """Placeholder strategies on the BTC validation split through the paper account: default limits at several costs, limits off, and buy and hold.
    Summary numbers only, no prices."""
    rows = []
    for name, fn in STRATEGIES.items():
        for scenario, limits, fee in [("your limits", LIMITS, c) for c in COSTS_BPS] + [("limits effectively off", LIMITS_OFF, 10)]:
            res = run_paper(candles, fn, limits=RiskLimits(**limits), fee_bps=fee, slippage_bps=0.0)
            rows.append({"strategy": name, "scenario": scenario, "fee_bps": fee, **{k: res.metrics[k] for k in METRIC_KEYS}})
    bench = buy_and_hold_equity(candles)
    ppy = res.metrics["periods_per_year"]
    ret = bench.pct_change().fillna(bench.iloc[0] - 1.0).to_numpy()
    rows.append({"strategy": "buy and hold (no costs)", "scenario": "benchmark", "fee_bps": 0, "net_return": float(bench.iloc[-1] - 1), "sharpe": sharpe_ratio(ret, ppy),
                 "max_drawdown": float(equity_and_drawdown(pd.DataFrame({"equity": bench.to_numpy()}))["drawdown"].min()), "exposure": 1.0})
    return pd.DataFrame(rows)


def git_head() -> tuple[str, bool]:
    def git(*args):
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10)

    head = git("rev-parse", "--short", "HEAD").stdout.strip() or "unknown"
    return head, bool(git("status", "--porcelain", "--", "src", "scripts", "dashboard").stdout.strip())


def main() -> None:
    wanted = sorted({f for e in EXPERIMENTS for f in e["results"]} | set(EXTRA_RESULTS))
    missing = [f for f in wanted if not (OUT / f).exists()]
    if missing:
        sys.exit(f"Missing results in outputs/: {missing}. Run the experiments (or scripts/export_equity_curves.py) first.")
    btc = DATA / "BTCUSDT_1h.csv"
    if not btc.exists():
        sys.exit("data/BTCUSDT_1h.csv is needed for the split dates and the paper-account examples (scripts/fetch_binance.py).")

    PUBLISHED.mkdir(exist_ok=True)
    for old in PUBLISHED.glob("*.csv"):
        old.unlink()
    manifest_files = {}
    for name in wanted:  # copied byte for byte
        shutil.copyfile(OUT / name, PUBLISHED / name)
        manifest_files[name] = {"generated_utc": utc((OUT / name).stat().st_mtime), "source": "outputs/ (unchanged copy)"}

    candles = load_candles(btc)
    derived = {
        "datasets_inventory.csv": dataset_inventory(DATA),
        "splits.csv": split_table(candles["timestamps"]),
        "paper_account_examples.csv": paper_examples(validation_candles(candles)),
    }
    now = utc(datetime.now(tz=timezone.utc).timestamp())
    for name, table in derived.items():
        table.to_csv(PUBLISHED / name, index=False)
        manifest_files[name] = {"generated_utc": now, "source": "derived summary (no raw prices)"}
    for name in manifest_files:
        manifest_files[name].update(sha256=sha256(PUBLISHED / name), bytes=(PUBLISHED / name).stat().st_size)

    head, dirty = git_head()
    manifest = {"published_utc": now, "source_commit": head + (" (+ uncommitted changes)" if dirty else ""),
                "note": "Result CSVs are unchanged copies of outputs/. Derived files contain summary numbers only; raw exchange candles are not published.",
                "files": dict(sorted(manifest_files.items()))}
    (PUBLISHED / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"published {len(manifest_files)} files to {PUBLISHED.relative_to(ROOT)}/ ({sum(f['bytes'] for f in manifest_files.values()) / 1024:.0f} KB), source commit {manifest['source_commit']}")


if __name__ == "__main__":
    main()
