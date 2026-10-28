"""Pre-declared grid: does ANY inference setting give Kronos-small measurable skill on BTC 1h validation windows?

DECLARED BEFORE RUNNING (do not edit after seeing results):
  variants (one change at a time from V0): V0 baseline, V1 greedy, V2 T=0.5, V3 lookback 128, V4 horizon 4
  windows: 300 non-overlapping, evenly spaced, validation split only; test split untouched
  primary criterion: rank correlation (IC) between predicted and actual end-of-horizon return, 99% bootstrap CI
                     lower bound > 0  (99% because 5 variants are tried: 5% split across 5 tries)
  secondary: direction accuracy vs the base rate of up-moves; excess error vs random walk (99% paired bootstrap CI)
  a passing variant must then be replicated unchanged on the HK sample before it is believed
  every variant is reported, including failures
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kronosquant.baselines import BASELINES  # noqa: E402
from kronosquant.cache import ForecastCache, run_walk_forward  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.evaluation import actual_paths, baseline_paths, paths_from_forecasts, select_windows  # noqa: E402
from kronosquant.forecast import ROOT, forecast_config, forecast_window, load_predictor  # noqa: E402
from kronosquant.metrics import bootstrap_ic_ci, directional_accuracy, end_mae, information_coefficient, paired_bootstrap_ci, path_mae  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402

BASE = dict(lookback=400, pred_len=12, T=1.0, top_p=0.9, top_k=0, sample_count=5)
GRID = {
    "V0_baseline": dict(BASE),
    "V1_greedy": {**BASE, "top_k": 1, "top_p": 1.0, "sample_count": 1},  # deterministic: no randomness left to average
    "V2_T0.5": {**BASE, "T": 0.5},
    "V3_lookback128": {**BASE, "lookback": 128},
    "V4_horizon4": {**BASE, "pred_len": 4},
}
LEVEL = 0.99

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/BTCUSDT_1h.csv")
parser.add_argument("--bar", default="1h")
parser.add_argument("--windows", type=int, default=300)
parser.add_argument("--variants", nargs="*", default=list(GRID))
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

candles = load_candles(ROOT / args.csv)
split = chronological_split(len(candles))["val"]
print(f"{len(GRID)} variants declared; running {args.variants}; {args.windows} windows each; level {LEVEL:.0%}", flush=True)

predictor = None
rows = []
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
for name in args.variants:
    cfg = GRID[name]
    windows = select_windows(candles, split, cfg["lookback"], cfg["pred_len"], args.windows, args.bar)
    config = forecast_config(candles, args.device, cfg["lookback"], cfg["pred_len"], cfg["T"], cfg["top_p"], cfg["sample_count"], cfg["top_k"])
    cache = ForecastCache(out / "cache", config)

    def forecast_one(w, cfg=cfg):
        global predictor
        if predictor is None:
            predictor = load_predictor(args.device)
        return forecast_window(predictor, candles, w.origin, w.lookback, w.pred_len, seed=w.origin, T=cfg["T"], top_p=cfg["top_p"], top_k=cfg["top_k"], sample_count=cfg["sample_count"])

    print(f"\n== {name}: {cfg}", flush=True)
    t0 = time.time()
    forecasts = run_walk_forward(windows, cache, forecast_one, progress=lambda m: print(" ", m, flush=True))
    k = paths_from_forecasts(candles, windows, forecasts)
    actual = actual_paths(candles, windows)
    rw = baseline_paths(candles, windows, BASELINES["random_walk"])
    mom = baseline_paths(candles, windows, BASELINES["momentum"])

    acc, n_dir = directional_accuracy(k[:, -1], actual[:, -1])
    ic = information_coefficient(k[:, -1], actual[:, -1])
    ic_lo, ic_hi = bootstrap_ic_ci(k[:, -1], actual[:, -1], level=LEVEL)
    err_k, err_rw = np.abs(k - actual).mean(axis=1), np.abs(rw - actual).mean(axis=1)
    mae_lo, mae_hi = paired_bootstrap_ci(err_k - err_rw, level=LEVEL)
    rows.append(
        {
            "variant": name, "n": len(windows), "minutes": round((time.time() - t0) / 60, 1),
            "IC": ic, "IC_lo99": ic_lo, "IC_hi99": ic_hi, "PASS_primary": bool(ic_lo > 0),
            "dir_acc": acc, "dir_n": n_dir, "base_rate_up": float((actual[:, -1] > 0).mean()),
            "path_MAE": path_mae(k, actual), "rw_path_MAE": path_mae(rw, actual),
            "excess_err_lo99": mae_lo, "excess_err_hi99": mae_hi,
            "corr_with_momentum": float(np.corrcoef(k[:, -1], mom[:, -1])[0, 1]),
        }
    )
    pd.DataFrame(rows).to_csv(out / "grid_BTCUSDT_1h.csv", index=False)  # saved after every variant
    np.savez(out / f"grid_{name}_paths.npz", actual=actual, kronos=k, origins=[w.origin for w in windows])
    r = rows[-1]
    print(f"  IC {r['IC']:+.3f} [{r['IC_lo99']:+.3f}, {r['IC_hi99']:+.3f}]  dir {r['dir_acc']:.1%} (n={r['dir_n']}, up-rate {r['base_rate_up']:.1%})  "
          f"excess err vs RW [{mae_lo:+.5f}, {mae_hi:+.5f}]  primary pass: {r['PASS_primary']}", flush=True)

pd.set_option("display.width", 250)
print("\n", pd.DataFrame(rows).set_index("variant").to_string(float_format=lambda x: f"{x:.4f}"))
