"""Stage 3: Kronos vs simple baselines on forecast quality (validation split only; test is untouched)."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kronosquant.baselines import BASELINES  # noqa: E402
from kronosquant.cache import ForecastCache, run_walk_forward  # noqa: E402
from kronosquant.evaluation import actual_paths, baseline_paths, paths_from_forecasts, select_windows  # noqa: E402
from kronosquant.forecast import ROOT, forecast_config, forecast_window, load_predictor, load_sample_candles  # noqa: E402
from kronosquant.metrics import (  # noqa: E402
    directional_accuracy,
    end_mae,
    information_coefficient,
    paired_bootstrap_ci,
    path_mae,
)
from kronosquant.splits import chronological_split  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--windows", type=int, default=100)
parser.add_argument("--pred-len", type=int, default=12)
parser.add_argument("--lookback", type=int, default=400)
parser.add_argument("--sample-count", type=int, default=5)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

candles = load_sample_candles()
split = chronological_split(len(candles))["val"]
windows = select_windows(candles, split, args.lookback, args.pred_len, args.windows)
first, last = candles["timestamps"].iloc[windows[0].origin], candles["timestamps"].iloc[windows[-1].origin]
print(f"{len(windows)} validation windows, {first} .. {last}, horizon {args.pred_len} candles, lookback {args.lookback}")

actual = actual_paths(candles, windows)
paths = {name: baseline_paths(candles, windows, fn) for name, fn in BASELINES.items()}

T, TOP_P = 1.0, 0.9
config = forecast_config(candles, args.device, args.lookback, args.pred_len, T, TOP_P, args.sample_count)
cache = ForecastCache(ROOT / "outputs" / "cache", config)
print("cache:", cache.dir)
predictor = None


def forecast_one(w):
    global predictor
    if predictor is None:  # load the model only if something is actually missing from the cache
        predictor = load_predictor(args.device)
    return forecast_window(predictor, candles, w.origin, w.lookback, w.pred_len, seed=w.origin, T=T, top_p=TOP_P, sample_count=args.sample_count)


t0 = time.time()
forecasts = run_walk_forward(windows, cache, forecast_one)
paths["kronos"] = paths_from_forecasts(candles, windows, forecasts)
print(f"Kronos: {len(windows)} forecasts ready in {time.time() - t0:.0f}s")

rw_err = np.abs(paths["random_walk"] - actual).mean(axis=1)  # per-window mean abs error of the random walk
rows = []
for name, p in paths.items():
    acc, n_dir = directional_accuracy(p[:, -1], actual[:, -1])
    err = np.abs(p - actual).mean(axis=1)
    lo, hi = paired_bootstrap_ci(err - rw_err) if name != "random_walk" else (np.nan, np.nan)
    rows.append(
        {
            "model": name,
            "path_MAE": path_mae(p, actual),
            "end_MAE": end_mae(p, actual),
            "dir_acc": acc,
            "dir_n": n_dir,
            "IC": information_coefficient(p[:, -1], actual[:, -1]),
            "MAE-RW 95% CI lo": lo,
            "hi": hi,
        }
    )
table = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 200)
print(table.to_string(float_format=lambda x: f"{x:.6f}"))
print(f"\nBase rate: price ended the 1-hour window higher in {(actual[:, -1] > 0).mean():.1%} of windows")
print("MAE-RW CI: mean(model error - random-walk error) per window; entirely below 0 = better than random walk, straddling 0 = cannot tell.")

out = ROOT / "outputs"
out.mkdir(exist_ok=True)
table.to_csv(out / "stage3_table.csv")
np.savez(out / "stage3_paths.npz", actual=actual, origins=[w.origin for w in windows], **paths)
