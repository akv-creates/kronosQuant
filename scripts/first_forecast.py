"""Stage 1 experiment: one Kronos-small forecast, plotted against what really happened."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kronosquant.forecast import ROOT, forecast_window, load_predictor, load_sample_candles  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

LOOKBACK, PRED_LEN = 400, 120
candles = load_sample_candles()
origin = len(candles) - PRED_LEN  # most recent window in the file

t0 = time.time()
predictor = load_predictor(args.device)
print(f"loaded model on {args.device} in {time.time() - t0:.1f}s")

t0 = time.time()
pred = forecast_window(predictor, candles, origin, LOOKBACK, PRED_LEN, seed=args.seed)
print(f"forecast of {PRED_LEN} candles took {time.time() - t0:.1f}s")
print(pred.head())

actual = candles.iloc[origin : origin + PRED_LEN].set_index("timestamps")
hist = candles.iloc[origin - LOOKBACK : origin].set_index("timestamps")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(LOOKBACK), hist["close"].values, label="history", color="gray")
x_future = range(LOOKBACK, LOOKBACK + PRED_LEN)
ax.plot(x_future, actual["close"].values, label="actual", color="blue")
ax.plot(x_future, pred["close"].values, label="Kronos forecast", color="red")
ax.set_xlabel("candle index (not calendar time; overnight gaps are skipped)")
ax.set_ylabel("close")
ax.legend()
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
fig.savefig(out / f"first_forecast_{args.device}.png", dpi=120, bbox_inches="tight")
pred.to_csv(out / f"first_forecast_{args.device}.csv")
print("saved", out / f"first_forecast_{args.device}.png")
