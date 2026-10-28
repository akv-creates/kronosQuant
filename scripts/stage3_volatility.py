"""Does Kronos forecast how VOLATILE the next candles are (price range), even if it cannot forecast direction?

Reads the cached Stage 3 forecasts (no model run). Range = (high - low) / close per candle, averaged over the
forecast horizon. Because the cache stores the mean over sample paths, this is an expected-range estimate.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kronosquant.cache import ForecastCache, load_cached  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.evaluation import select_windows  # noqa: E402
from kronosquant.forecast import ROOT, forecast_config, load_sample_candles  # noqa: E402
from kronosquant.metrics import information_coefficient, log_ratio_mae, paired_bootstrap_ci  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--csv")
parser.add_argument("--bar", default="5min")
parser.add_argument("--windows", type=int, default=100)
parser.add_argument("--pred-len", type=int, default=12)
parser.add_argument("--lookback", type=int, default=400)
parser.add_argument("--recent", type=int, default=48, help="bars for the 'recent range' baseline")
args = parser.parse_args()

candles = load_candles(Path(args.csv)) if args.csv else load_sample_candles()
windows = select_windows(candles, chronological_split(len(candles))["val"], args.lookback, args.pred_len, args.windows, args.bar)
cache = ForecastCache(ROOT / "outputs" / "cache", forecast_config(candles, "cpu", args.lookback, args.pred_len, 1.0, 0.9, 5))
forecasts = load_cached(windows, cache)  # raises if the Stage 3 run has not been done for these settings

rng = (candles["high"] - candles["low"]) / candles["close"]
actual, kronos, hist_all, hist_recent, seasonal = [], [], [], [], []
slot = candles["timestamps"].dt.strftime("%H:%M")  # clock slot of each candle, e.g. 09:35
for w in windows:
    f = forecasts[w.origin]
    kronos.append(((f["high"] - f["low"]) / f["close"]).mean())
    actual.append(rng.iloc[w.future].mean())
    hist_all.append(rng.iloc[w.history].mean())
    hist_recent.append(rng.iloc[w.origin - args.recent : w.origin].mean())
    # Time-of-day baseline: for each future candle, the average range of history candles in the same clock slot.
    by_slot = rng.iloc[w.history].groupby(slot.iloc[w.history]).mean()
    seasonal.append(np.mean([by_slot.get(slot.iloc[i], hist_all[-1]) for i in w.future]))
actual, preds = np.array(actual), {"hist_all": np.array(hist_all), "hist_recent": np.array(hist_recent), "same_time_of_day": np.array(seasonal), "kronos": np.array(kronos)}

print(f"{len(windows)} windows; mean actual range {actual.mean() * 1e4:.1f} bp per candle")
rows = []
for name, p in preds.items():
    err = np.abs(np.log(p / actual))
    ref = np.abs(np.log(preds["hist_all"] / actual))
    lo, hi = paired_bootstrap_ci(err - ref) if name != "hist_all" else (np.nan, np.nan)
    rows.append({"model": name, "mean_pred_bp": p.mean() * 1e4, "log_ratio_MAE": log_ratio_mae(p, actual), "rank_IC": information_coefficient(p, actual), "vs hist_all CI lo": lo, "hi": hi})
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).set_index("model").to_string(float_format=lambda x: f"{x:.4f}"))
print("\nlog_ratio_MAE 0.2 ~ off by about 20% on average. rank_IC: do high forecasts coincide with truly volatile windows (noise ~ +-0.1).")
print("CI: mean(model error - hist_all error); entirely below 0 = better than the long-history average, straddling 0 = cannot tell.")

d = np.abs(np.log(preds["kronos"] / actual)) - np.abs(np.log(preds["same_time_of_day"] / actual))
lo, hi = paired_bootstrap_ci(d)
print(f"\nKronos minus same_time_of_day error: mean {d.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]  (above 0 = Kronos worse, straddling 0 = cannot tell)")
