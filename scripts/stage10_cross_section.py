"""Stage 10: does pretrained Kronos-small RANK crypto assets correctly at each date? (cross-sectional test, validation split only)

DECLARED BEFORE RUNNING (do not edit after seeing results):
  universe:   12 Binance USDT pairs fixed in advance: BTC ETH BNB XRP ADA DOGE LTC TRX LINK BCH ETC XLM, 1h, 3 years to 2026-10-28.
              All 12 were checked for full history and zero gaps before any forecast. Known flaw: the list is picked today, so
              assets that were delisted or collapsed are absent (survivorship bias). It can flatter any strategy and is not fixed here.
  dates:      100 non-overlapping, evenly spaced 12-hour windows from the validation split (same origins for every asset).
  forecasts:  pretrained Kronos-small, V0 settings from the Stage 3 grid (lookback 400, 12 bars, T 1.0, top_p 0.9, sample_count 5),
              seed = window origin. Fine-tuned models are not used (they were worse).
  primary:    mean over dates of the cross-asset rank correlation (Spearman) between predicted and actual 12-bar log-return;
              99% bootstrap CI over dates; PASS only if its lower bound is > 0.
  secondary (reported, not used to rescue a failed primary):
              same IC for the reversal baseline (minus past 12-bar return) and the momentum baseline; paired difference
              Kronos minus reversal (99% CI over dates); share of dates with IC > 0; IC after dividing predicted and actual
              returns by each asset's history volatility.
  after:      a PASS that also beats reversal would justify a cost-aware long/short test and then ONE look at the test split.
              A fail ends this idea for this model on this universe.
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
from kronosquant.crosssection import rank_ic_by_date  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.evaluation import actual_paths, baseline_paths, paths_from_forecasts, select_windows  # noqa: E402
from kronosquant.config import CONFIDENCE_LEVEL as LEVEL, KRONOS_V0 as V0, ROOT, UNIVERSE  # noqa: E402
from kronosquant.forecast import forecast_config, make_forecaster  # noqa: E402
from kronosquant.metrics import paired_bootstrap_ci  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--dates", type=int, default=100)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

data = {s: load_candles(ROOT / "data" / f"{s}_1h.csv") for s in UNIVERSE}
btc = data["BTCUSDT"]
for s, c in data.items():
    if not c["timestamps"].equals(btc["timestamps"]):
        sys.exit(f"{s}: timestamps differ from BTC; rows would not line up")
split = chronological_split(len(btc))["val"]
windows = select_windows(btc, split, V0["lookback"], V0["pred_len"], args.dates, "1h")
print(f"{len(UNIVERSE)} assets x {len(windows)} dates = {len(UNIVERSE) * len(windows)} forecasts; val rows {split.start}..{split.stop - 1}", flush=True)

pred_cols, actual_cols, mom_cols, vol_cols = [], [], [], []
for s in UNIVERSE:
    c = data[s]
    cache = ForecastCache(ROOT / "outputs" / "cache", forecast_config(c, args.device, V0["lookback"], V0["pred_len"], V0["T"], V0["top_p"], V0["sample_count"], V0["top_k"]))

    forecast_one = make_forecaster(c, args.device, T=V0["T"], top_p=V0["top_p"], sample_count=V0["sample_count"], top_k=V0["top_k"])

    print(f"== {s}", flush=True)
    t0 = time.time()
    forecasts = run_walk_forward(windows, cache, forecast_one, progress=lambda m: print(" ", m, flush=True))
    pred_cols.append(paths_from_forecasts(c, windows, forecasts)[:, -1])
    actual_cols.append(actual_paths(c, windows)[:, -1])
    mom_cols.append(baseline_paths(c, windows, BASELINES["momentum"])[:, -1])
    r = np.log(c["close"]).diff().to_numpy()
    vol_cols.append(np.array([np.nanstd(r[w.history.start + 1 : w.history.stop]) * np.sqrt(V0["pred_len"]) for w in windows]))  # history only
    print(f"   {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)

pred, actual, mom, vol = (np.stack(x, axis=1) for x in (pred_cols, actual_cols, mom_cols, vol_cols))  # (dates, assets)
np.savez(ROOT / "outputs" / "stage10_matrices.npz", pred=pred, actual=actual, mom=mom, vol=vol, origins=[w.origin for w in windows], assets=UNIVERSE)


def summarize(name, ics):
    ics = ics[~np.isnan(ics)]
    lo, hi = paired_bootstrap_ci(ics, level=LEVEL)
    return {"signal": name, "dates": len(ics), "mean_IC": ics.mean(), "ci_lo99": lo, "ci_hi99": hi, "share_dates_pos": float((ics > 0).mean()), "PASS_primary": bool(lo > 0)}


ic_k = rank_ic_by_date(pred, actual)
ic_rev = rank_ic_by_date(-mom, actual)
ic_mom = rank_ic_by_date(mom, actual)
ic_k_vol = rank_ic_by_date(pred / vol, actual / vol)
rows = [summarize("kronos", ic_k), summarize("reversal_baseline", ic_rev), summarize("momentum_baseline", ic_mom), summarize("kronos_volscaled", ic_k_vol)]
d_lo, d_hi = paired_bootstrap_ci(ic_k - ic_rev, level=LEVEL)
table = pd.DataFrame(rows)
table.to_csv(ROOT / "outputs" / "stage10_cross_section.csv", index=False)
pd.set_option("display.width", 200)
print("\n", table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print(f"\nKronos minus reversal IC: mean {np.mean(ic_k - ic_rev):+.4f}, {LEVEL:.0%} CI [{d_lo:+.4f}, {d_hi:+.4f}]")
print(f"corr(Kronos, momentum) across assets, averaged over dates: {np.nanmean(rank_ic_by_date(pred, mom)):+.3f}")
print(f"PRIMARY (Kronos mean cross-sectional IC, {LEVEL:.0%} CI lower > 0): {rows[0]['PASS_primary']}")
