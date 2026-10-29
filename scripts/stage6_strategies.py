"""Stage 6: do the cached Kronos forecasts (V0, BTC 1h, validation split) make money after costs?

DECLARED BEFORE RUNNING (do not edit after seeing results):
  forecasts: V0 baseline from the Stage 3 grid, read from the cache (no model is run). One forecast per window,
             12-bar horizon, position held for those 12 bars, flat between windows. Validation split only; test untouched.
  primary:   Kronos long/short (sign of the forecast end-of-window return) at 10 bps one-way cost.
             PASS only if the lower end of the 99% bootstrap CI of the mean per-window net return is > 0.
  not done:  we do NOT flip the sign. The Stage 3 IC was not significantly negative, so "trade the opposite" would be a fit to noise.
  secondary (reported, never used to rescue a failed primary): long-only Kronos, momentum, contrarian (opposite of momentum),
             always-long inside the windows, flat, buy-and-hold over the whole split; costs 0/5/10/20 bps;
             a null of 1000 random long/short coin-flip strategies at 10 bps (share that did at least as well as Kronos).
  costs:     "bps" is the all-in one-way cost per unit of position change (fee_bps = bps, slippage_bps = 0).
             Real Binance fees and slippage are NOT verified.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kronosquant.backtest import run_backtest  # noqa: E402
from kronosquant.baselines import BASELINES  # noqa: E402
from kronosquant.cache import ForecastCache, load_cached  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.evaluation import baseline_paths, paths_from_forecasts, select_windows  # noqa: E402
from kronosquant.forecast import ROOT, forecast_config  # noqa: E402
from kronosquant.metrics import paired_bootstrap_ci  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402
from kronosquant.strategy import directions_from_forecast, window_returns, window_signal  # noqa: E402

V0 = dict(lookback=400, pred_len=12, T=1.0, top_p=0.9, top_k=0, sample_count=5)  # same values as stage3_grid.py V0_baseline
COSTS = [0, 5, 10, 20]
PRIMARY_COST = 10
LEVEL = 0.99
N_NULL = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/BTCUSDT_1h.csv")
parser.add_argument("--bar", default="1h")
parser.add_argument("--windows", type=int, default=300)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

candles = load_candles(ROOT / args.csv)
split = chronological_split(len(candles))["val"]
H = V0["pred_len"]
windows = select_windows(candles, split, V0["lookback"], H, args.windows, args.bar)
cache = ForecastCache(ROOT / "outputs" / "cache", forecast_config(candles, args.device, V0["lookback"], H, V0["T"], V0["top_p"], V0["sample_count"], V0["top_k"]))
forecasts = load_cached(windows, cache)  # raises if the grid has not been run
pred_end = paths_from_forecasts(candles, windows, forecasts)[:, -1]
mom_end = baseline_paths(candles, windows, BASELINES["momentum"])[:, -1]

# Work on the validation rows only. Keep windows whose entry signal bar (origin - 1) and exit bar (origin + H) are both
# inside them; a window starting on the first validation row would need its signal on a training row, so it is dropped.
val = candles.iloc[split.start : split.stop].reset_index(drop=True)
keep = [i for i, w in enumerate(windows) if 1 <= w.origin - split.start and w.origin - split.start + H < len(val)]
origins = [windows[i].origin - split.start for i in keep]
pred_end, mom_end = pred_end[keep], mom_end[keep]
print(f"{len(origins)} windows of {H} bars inside {len(val)} validation bars ({val['timestamps'].iloc[0]} .. {val['timestamps'].iloc[-1]})", flush=True)

# name -> (directions per window, allow_short)
n = len(origins)
strategies = {
    "kronos_longshort": (directions_from_forecast(pred_end, 0.0, True), True),
    "kronos_longonly": (directions_from_forecast(pred_end, 0.0, False), False),
    "momentum_longshort": (directions_from_forecast(mom_end, 0.0, True), True),
    "contrarian_longshort": (-directions_from_forecast(mom_end, 0.0, True), True),
    "always_long_in_windows": (np.ones(n), False),
    "flat": (np.zeros(n), False),
}


def run(directions, allow_short, cost):
    sig = window_signal(len(val), origins, directions, H)
    return run_backtest(val, sig, fee_bps=cost, slippage_bps=0, allow_short=allow_short)


rows, per_window = [], {}
for name, (dirs, short) in strategies.items():
    for cost in COSTS:
        r = run(dirs, short, cost)
        w = window_returns(r.frame["net_ret"].to_numpy(), origins, H)
        lo, hi = paired_bootstrap_ci(w, level=LEVEL) if w.std() > 0 else (0.0, 0.0)
        per_window[(name, cost)] = w
        m = r.metrics
        rows.append({"strategy": name, "cost_bps": cost, "mean_window_bps": w.mean() * 1e4, "ci_lo99_bps": lo * 1e4, "ci_hi99_bps": hi * 1e4,
                     "net_return": m["net_return"], "gross_return": m["gross_return"], "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"],
                     "turnover_total": m["turnover_total"], "exposure": m["exposure"], "hit_windows": float((w > 0).mean())})

bh = run_backtest(val, pd.Series(1.0, index=val.index), fee_bps=PRIMARY_COST, slippage_bps=0)
rows.append({"strategy": "buy_hold_whole_split", "cost_bps": PRIMARY_COST, "net_return": bh.metrics["net_return"], "gross_return": bh.metrics["gross_return"],
             "sharpe": bh.metrics["sharpe"], "max_drawdown": bh.metrics["max_drawdown"], "turnover_total": bh.metrics["turnover_total"], "exposure": bh.metrics["exposure"]})

# Null: coin-flip long/short strategies on the same windows and costs.
kronos_net = next(r["net_return"] for r in rows if r["strategy"] == "kronos_longshort" and r["cost_bps"] == PRIMARY_COST)
rng = np.random.default_rng(0)
null_net = np.array([run(rng.choice([-1.0, 1.0], size=n), True, PRIMARY_COST).metrics["net_return"] for _ in range(N_NULL)])
share_at_least = float((null_net >= kronos_net).mean())

table = pd.DataFrame(rows)
out = ROOT / "outputs"
table.to_csv(out / "stage6_strategies.csv", index=False)
pd.set_option("display.width", 250)
print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"\nrandom long/short null at {PRIMARY_COST} bps: net return 5th/50th/95th pct = "
      f"{np.percentile(null_net, 5):+.3f} / {np.percentile(null_net, 50):+.3f} / {np.percentile(null_net, 95):+.3f}; "
      f"Kronos {kronos_net:+.3f}; share of random strategies at least as good: {share_at_least:.3f}")
p = next(r for r in rows if r["strategy"] == "kronos_longshort" and r["cost_bps"] == PRIMARY_COST)
print(f"PRIMARY (Kronos long/short, {PRIMARY_COST} bps): mean per-window net {p['mean_window_bps']:+.1f} bps, "
      f"{LEVEL:.0%} CI [{p['ci_lo99_bps']:+.1f}, {p['ci_hi99_bps']:+.1f}] -> pass: {p['ci_lo99_bps'] > 0}")
