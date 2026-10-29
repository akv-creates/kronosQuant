"""Stage 12: was Kronos-small's output drowned in sampling noise? Same settings, 20 sampled paths instead of 5 (BTC 1h validation only).

DECLARED BEFORE RUNNING (do not edit after seeing results):
  why:        the forecast is the average of a few randomly sampled paths. If the model carries a faint signal, averaging more paths
              should reveal it. The Stage 3 grid never changed sample_count.
  windows:    ALL non-overlapping 12-hour windows in the BTC 1h validation split (438; selected by rule, not by results).
              Stage 3 used 300 of them evenly spaced; the other 138 are new to every setting. Test split untouched.
  settings:   V0 from the Stage 3 grid (lookback 400, 12 bars, T 1.0, top_p 0.9, top_k 0), seed = window origin.
              S5 = sample_count 5 (the earlier setting; 300 windows already cached), S20 = sample_count 20.
  primary:    S20 rank correlation (IC) between predicted and actual end-of-window return; 99% bootstrap CI over windows;
              PASS only if the lower bound is > 0. (99% to stay consistent with the Stage 3 grid, since this adds to the same
              family of tries.) Power note: with about 438 windows the 99% interval is roughly +-0.12 wide, so a pass needs an
              observed IC of about 0.12 or more. An IC below that cannot be detected here whatever the noise level.
  secondary (reported, not used to rescue a failed primary):
              IC of S5 on the same windows; paired difference IC(S20) - IC(S5) with 99% CI over windows; correlation between the
              S5 and S20 end-of-window forecasts (low = the 5-path forecast was mostly noise; high = more paths changed little);
              direction accuracy vs the base rate of up-moves; excess path error vs the random walk (99% paired CI);
              correlation of the forecast with the momentum baseline.
  after:      a PASS must be replicated on the 12-asset cross-section with 20 paths before it is believed, then ONE look at the
              test split. A fail ends sampling noise as an explanation for this model on this asset.
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
from kronosquant.metrics import bootstrap_ic_ci, directional_accuracy, information_coefficient, paired_bootstrap_ci, paired_ic_diff_ci, path_mae  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402

BASE = dict(lookback=400, pred_len=12, T=1.0, top_p=0.9, top_k=0)
SETTINGS = {"S5": {**BASE, "sample_count": 5}, "S20": {**BASE, "sample_count": 20}}
LEVEL = 0.99

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

candles = load_candles(ROOT / "data" / "BTCUSDT_1h.csv")
split = chronological_split(len(candles))["val"]
windows = select_windows(candles, split, BASE["lookback"], BASE["pred_len"], 10_000, "1h")  # n larger than available: keeps all
print(f"{len(windows)} validation windows, level {LEVEL:.0%}", flush=True)

predictor = None
paths = {}
for name, cfg in SETTINGS.items():
    cache = ForecastCache(ROOT / "outputs" / "cache", forecast_config(candles, args.device, cfg["lookback"], cfg["pred_len"], cfg["T"], cfg["top_p"], cfg["sample_count"], cfg["top_k"]))

    def forecast_one(w, cfg=cfg):
        global predictor
        if predictor is None:
            predictor = load_predictor(args.device)
        return forecast_window(predictor, candles, w.origin, w.lookback, w.pred_len, seed=w.origin, T=cfg["T"], top_p=cfg["top_p"], top_k=cfg["top_k"], sample_count=cfg["sample_count"])

    print(f"\n== {name}: {cfg}", flush=True)
    t0 = time.time()
    forecasts = run_walk_forward(windows, cache, forecast_one, progress=lambda m: print(" ", m, flush=True))
    paths[name] = paths_from_forecasts(candles, windows, forecasts)
    print(f"   {name} ready in {(time.time() - t0) / 60:.1f} min", flush=True)

actual = actual_paths(candles, windows)
rw = baseline_paths(candles, windows, BASELINES["random_walk"])
mom = baseline_paths(candles, windows, BASELINES["momentum"])
out = ROOT / "outputs"
np.savez(out / "stage12_paths.npz", actual=actual, S5=paths["S5"], S20=paths["S20"], origins=[w.origin for w in windows])

rows = []
for name, k in paths.items():
    acc, n_dir = directional_accuracy(k[:, -1], actual[:, -1])
    ic = information_coefficient(k[:, -1], actual[:, -1])
    lo, hi = bootstrap_ic_ci(k[:, -1], actual[:, -1], level=LEVEL)
    err_k, err_rw = np.abs(k - actual).mean(axis=1), np.abs(rw - actual).mean(axis=1)
    e_lo, e_hi = paired_bootstrap_ci(err_k - err_rw, level=LEVEL)
    rows.append({"setting": name, "sample_count": SETTINGS[name]["sample_count"], "n": len(windows), "IC": ic, "IC_lo99": lo, "IC_hi99": hi,
                 "PASS_primary": bool(lo > 0) if name == "S20" else np.nan, "dir_acc": acc, "base_rate_up": float((actual[:, -1] > 0).mean()),
                 "path_MAE": path_mae(k, actual), "rw_path_MAE": path_mae(rw, actual), "excess_err_lo99": e_lo, "excess_err_hi99": e_hi,
                 "corr_with_momentum": float(np.corrcoef(k[:, -1], mom[:, -1])[0, 1])})
table = pd.DataFrame(rows)
table.to_csv(out / "stage12_samples.csv", index=False)
pd.set_option("display.width", 250)
print("\n", table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

d_lo, d_hi = paired_ic_diff_ci(paths["S20"][:, -1], paths["S5"][:, -1], actual[:, -1], level=LEVEL)
ic20, ic5 = rows[1]["IC"], rows[0]["IC"]
pear = float(np.corrcoef(paths["S5"][:, -1], paths["S20"][:, -1])[0, 1])
spear = information_coefficient(paths["S5"][:, -1], paths["S20"][:, -1])
print(f"\nIC(S20) - IC(S5) = {ic20 - ic5:+.4f}, {LEVEL:.0%} CI [{d_lo:+.4f}, {d_hi:+.4f}]")
print(f"correlation of S5 and S20 end-of-window forecasts: Pearson {pear:+.3f}, Spearman {spear:+.3f}")
print(f"half-width of the S20 IC interval: {(rows[1]['IC_hi99'] - rows[1]['IC_lo99']) / 2:.3f} (an observed IC below this cannot pass)")
print(f"PRIMARY (S20 IC, {LEVEL:.0%} CI lower > 0): {rows[1]['PASS_primary']}")
