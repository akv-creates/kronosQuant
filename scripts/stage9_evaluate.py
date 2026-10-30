"""Stage 9b: does the fine-tuned checkpoint forecast better than the pretrained model? (BTC 1h, validation split)

DECLARED BEFORE EVALUATING (do not edit after seeing results):
  candidate:  the single checkpoint saved by scripts/stage9_finetune.py (lowest tune loss). No other checkpoint is evaluated.
  settings:   V0 from the Stage 3 grid (lookback 400, 12 bars, T 1.0, top_p 0.9, sample_count 5), the same 300 validation windows.
  primary:    IC between predicted and actual 12-bar return, 99% bootstrap CI lower bound > 0  -> PASS.
  secondary (reported, not used to rescue a failed primary): direction accuracy vs base rate; excess error vs random walk
              (99% paired bootstrap); error of fine-tuned minus error of pretrained (99% paired bootstrap, same windows);
              validation token loss of both models (a likelihood diagnostic, not skill); correlation with momentum.
  next:       only a PASS leads to touching the test split, once, with this same frozen checkpoint.
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
from kronosquant.finetune import evaluate_loss, prepare_arrays, window_starts  # noqa: E402
from kronosquant.config import CONFIDENCE_LEVEL as LEVEL, KRONOS_V0 as V0, ROOT  # noqa: E402
from kronosquant.forecast import forecast_config, make_forecaster  # noqa: E402
from kronosquant.metrics import bootstrap_ic_ci, directional_accuracy, information_coefficient, paired_bootstrap_ci, path_mae  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="data/BTCUSDT_1h.csv")
parser.add_argument("--bar", default="1h")
parser.add_argument("--checkpoint", default="outputs/finetune/btc1h_v1/best")
parser.add_argument("--windows", type=int, default=300)
parser.add_argument("--device", default="cpu")
parser.add_argument("--split", default="val", choices=["val", "test"])
args = parser.parse_args()
if args.split == "test":
    print("TEST SPLIT: allowed only after a PASS on validation, once, with the frozen checkpoint.", flush=True)

candles = load_candles(ROOT / args.csv)
split = chronological_split(len(candles))[args.split]
ckpt = ROOT / args.checkpoint
windows = select_windows(candles, split, V0["lookback"], V0["pred_len"], args.windows, args.bar)
config = forecast_config(candles, args.device, V0["lookback"], V0["pred_len"], V0["T"], V0["top_p"], V0["sample_count"], V0["top_k"], checkpoint_dir=ckpt)
cache = ForecastCache(ROOT / "outputs" / "cache", config)
forecast_one = make_forecaster(candles, args.device, T=V0["T"], top_p=V0["top_p"], sample_count=V0["sample_count"], top_k=V0["top_k"], checkpoint_dir=ckpt)


t0 = time.time()
forecasts = run_walk_forward(windows, cache, forecast_one, progress=lambda m: print(" ", m, flush=True))
k = paths_from_forecasts(candles, windows, forecasts)
actual = actual_paths(candles, windows)
rw = baseline_paths(candles, windows, BASELINES["random_walk"])
mom = baseline_paths(candles, windows, BASELINES["momentum"])

ic = information_coefficient(k[:, -1], actual[:, -1])
ic_lo, ic_hi = bootstrap_ic_ci(k[:, -1], actual[:, -1], level=LEVEL)
acc, n_dir = directional_accuracy(k[:, -1], actual[:, -1])
err_k, err_rw = np.abs(k - actual).mean(axis=1), np.abs(rw - actual).mean(axis=1)
rw_lo, rw_hi = paired_bootstrap_ci(err_k - err_rw, level=LEVEL)
row = {"split": args.split, "n": len(windows), "minutes": round((time.time() - t0) / 60, 1), "IC": ic, "IC_lo99": ic_lo, "IC_hi99": ic_hi, "PASS_primary": bool(ic_lo > 0),
       "dir_acc": acc, "dir_n": n_dir, "base_rate_up": float((actual[:, -1] > 0).mean()), "path_MAE": path_mae(k, actual), "rw_path_MAE": path_mae(rw, actual),
       "excess_err_vs_rw_lo99": rw_lo, "excess_err_vs_rw_hi99": rw_hi, "corr_with_momentum": float(np.corrcoef(k[:, -1], mom[:, -1])[0, 1])}

# Pretrained comparison on the same windows (paths saved by the grid run), validation only.
base_file = ROOT / "outputs" / "grid_V0_baseline_paths.npz"
if args.split == "val" and base_file.exists():
    base = np.load(base_file)
    if np.array_equal(base["origins"], [w.origin for w in windows]):
        err_pre = np.abs(base["kronos"] - actual).mean(axis=1)
        d_lo, d_hi = paired_bootstrap_ci(err_k - err_pre, level=LEVEL)
        row.update({"pretrained_IC": information_coefficient(base["kronos"][:, -1], actual[:, -1]), "err_vs_pretrained_lo99": d_lo, "err_vs_pretrained_hi99": d_hi})

# Token loss on validation windows for the pretrained and the fine-tuned predictor (diagnostic only).
if args.split == "val":
    from kronosquant.forecast import MODEL_ID, MODEL_REVISION, TOKENIZER_ID, TOKENIZER_REVISION  # noqa: E402
    from model import Kronos, KronosTokenizer  # noqa: E402

    tok = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION).to(args.device).eval()
    values, stamps = prepare_arrays(candles)
    starts = window_starts(split, 412, 206)
    for name, m in (("pretrained", Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION)), ("finetuned", Kronos.from_pretrained(str(ckpt)))):
        row[f"val_token_loss_{name}"] = evaluate_loss(m.to(args.device), tok, values, stamps, starts, 412, args.device)

out = ROOT / "outputs"
pd.DataFrame([row]).to_csv(out / f"stage9_evaluate_{args.split}.csv", index=False)
for key, v in row.items():
    print(f"  {key}: {v:.5f}" if isinstance(v, float) else f"  {key}: {v}")
