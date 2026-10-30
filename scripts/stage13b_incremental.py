"""Stage 13b: does Kronos add anything on top of the one baseline that passed (XSMOM_28)? (daily bars, validation periods only)

DECLARED BEFORE RUNNING (do not edit after seeing results):
  why:        Stage 13a found one marginal pass, cross-sectional 28-day momentum. The product question is whether Kronos adds value
              ON TOP of it, not whether Kronos beats nothing.
  contamination: Kronos was pretrained on exchange data that may include 2019-2025 (the corpus is not disclosed per date), so
              the primary window may not be out of sample for the model. A PASS is therefore an UPPER bound on what Kronos could
              do and is not evidence of a usable edge. Only a FAIL or an INCONCLUSIVE result is informative.
  data:       the same 12 daily pairs as Stage 13a, rows dated 2026-03-23 or later are never loaded (test period untouched).
  schedule:   the Stage 13a 5-day schedule (first decision 2018-12-31, traded at the next open, held 5 days), but only EVERY SECOND
              decision is used (about 240 periods) to halve the compute; this is a compute decision, not a result-driven one.
              Periods must start on or after 2019-01-01 and end on or before 2025-08-15 (same primary window as 13a).
  forecasts:  pretrained Kronos-small, daily bars, lookback 120, 5 bars ahead, T 1.0, top_p 0.9, 5 sampled paths, CPU, seed = window origin.
              Nothing is tuned. Score = log(forecast close of bar 5 / last close). An asset is eligible once it has 120 bars of history.
  arms (all dollar neutral, top third long / bottom third short, as in 13a; same eligible assets in every arm):
              BASE  XSMOM_28 restricted to the eligible assets;
              COMB  equal-weight blend of the cross-sectional percentile ranks of XSMOM_28 and the Kronos score (portfolio.rank_blend);
              KRON  Kronos score alone.
  primary:    mean over periods of net(COMB) - net(BASE) at 10 bps one-way (assumed, not a verified cost); 99% stationary block
              bootstrap, mean block 2 periods (about the same calendar span as the 4-period blocks of 13a), 5000 resamples, seed 0.
              PASS = lower end above 0; FAILED = upper end below 0; INCONCLUSIVE = interval includes 0. The half-width is printed.
  cost note:  between evaluated periods the engine assumes positions are held through the skipped 5 days, so turnover and cost are
              understated against a real half schedule. The primary is a paired difference, so the bias is shared; a secondary row
              charges a full round trip (flat between periods) to show how much it matters.
  secondary (reported, never used to rescue a failed primary): net mean and interval of each arm; costs 0 and 20 bps for the paired
              difference; round-trip cost variant; rank IC by date of KRON, XSMOM_28 and COMB against the realised open-to-open
              return, with the paired IC difference COMB - BASE (99% interval, block bootstrap over dates); mean rank correlation
              between the Kronos and momentum scores; per-year paired difference.
  after:      a PASS justifies only a forward-only (live, paper) record of the blend; it does NOT justify opening the test split, because
              of the contamination caveat. A FAIL or INCONCLUSIVE result ends this idea for this model at this horizon.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kronosquant.cache import ForecastCache, run_walk_forward  # noqa: E402
from kronosquant.crosssection import rank_ic_by_date  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.config import ROOT, UNIVERSE  # noqa: E402
from kronosquant.forecast import forecast_config, make_forecaster  # noqa: E402
from kronosquant.metrics import stationary_bootstrap_ci  # noqa: E402
from kronosquant.portfolio import decision_indices, period_stats, rank_blend, run_portfolio, trailing_return, xs_weights  # noqa: E402
from kronosquant.splits import Window  # noqa: E402

TEST_START = pd.Timestamp("2026-03-23")  # rows from here on are never loaded
FIRST_DECISION, STEP, EVERY, LEVEL, MEAN_BLOCK, N_BOOT, MIN_ASSETS = "2018-12-31", 5, 2, 0.99, 2.0, 5000, 6
WINDOW = (pd.Timestamp("2019-01-01"), pd.Timestamp("2025-08-15"))
PRIMARY_COST = 10
KRONOS = dict(lookback=120, pred_len=5, T=1.0, top_p=0.9, top_k=0, sample_count=5)
PPY = 365.0 / (STEP * EVERY)

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
parser.add_argument("--count", action="store_true", help="print how many forecasts are needed and stop")
parser.add_argument("--limit", type=int, default=0, help="smoke test: compute at most this many missing forecasts, then stop without analysis")
args = parser.parse_args()

frames = {}
for s in UNIVERSE:
    c = load_candles(ROOT / "data" / f"{s}_1d.csv")
    frames[s] = c[c["timestamps"] < TEST_START].reset_index(drop=True)  # hard cut: the test period is never read
dates = pd.DatetimeIndex(frames["BTCUSDT"]["timestamps"])
open_ = pd.DataFrame({s: f.set_index("timestamps")["open"].reindex(dates) for s, f in frames.items()})
close = pd.DataFrame({s: f.set_index("timestamps")["close"].reindex(dates) for s, f in frames.items()})
all_decisions = decision_indices(dates, FIRST_DECISION, STEP)[::EVERY]
decisions = [d for d in all_decisions if d + 1 + STEP < len(dates) and dates[d + 1] >= WINDOW[0] and dates[d + 1 + STEP] <= WINDOW[1]]
print(f"{len(decisions)} periods {dates[decisions[0] + 1]:%Y-%m-%d} .. {dates[decisions[-1] + 1 + STEP]:%Y-%m-%d}", flush=True)

# which (asset, origin) forecasts are needed: the asset has a bar on the origin day and at least `lookback` bars before it
need = {}
for s in UNIVERSE:
    pos = pd.Series(np.arange(len(frames[s])), index=pd.DatetimeIndex(frames[s]["timestamps"]))
    origins = [int(pos[dates[d + 1]]) for d in decisions if dates[d + 1] in pos.index and pos[dates[d + 1]] >= KRONOS["lookback"]]
    need[s] = origins
total = sum(len(v) for v in need.values())
print(f"forecasts needed: {total} ({', '.join(f'{s[:-4]} {len(v)}' for s, v in need.items())})", flush=True)
if args.count:
    sys.exit(0)

budget = args.limit
for s in UNIVERSE:
    c = frames[s]
    cache = ForecastCache(ROOT / "outputs" / "cache", forecast_config(c, args.device, KRONOS["lookback"], KRONOS["pred_len"], KRONOS["T"], KRONOS["top_p"], KRONOS["sample_count"], KRONOS["top_k"]))
    windows = [Window(o, KRONOS["lookback"], KRONOS["pred_len"]) for o in need[s]]
    if args.limit:
        windows = [w for w in windows if cache.has(w.origin)] + [w for w in windows if not cache.has(w.origin)][: max(0, budget)]
        windows.sort(key=lambda w: w.origin)
        budget -= sum(1 for w in windows if not cache.has(w.origin))

    forecast_one = make_forecaster(c, args.device, T=KRONOS["T"], top_p=KRONOS["top_p"], sample_count=KRONOS["sample_count"], top_k=KRONOS["top_k"])

    print(f"== {s}", flush=True)
    t0 = time.time()
    fc = run_walk_forward(windows, cache, forecast_one, progress=lambda m: print(" ", m, flush=True))
    if args.limit:
        if budget <= 0:
            print(f"smoke test done after {time.time() - t0:.0f}s on the last asset; stopping before analysis")
            sys.exit(0)
        continue
    print(f"   {s} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    need[s] = {w.origin: float(np.log(fc[w.origin]["close"].iloc[-1] / c["close"].iloc[w.origin - 1])) for w in windows}
if args.limit:
    sys.exit(0)

# ---- analysis (reads only decided-by-close-of-d information for scores; realised returns come from the engine's open-to-open convention)
kron = pd.DataFrame(np.nan, index=dates, columns=UNIVERSE)
for s in UNIVERSE:
    pos = pd.Series(np.arange(len(frames[s])), index=pd.DatetimeIndex(frames[s]["timestamps"]))
    for d in decisions:
        o = pos.get(dates[d + 1])
        if o is not None and o in need[s]:
            kron.iloc[d, kron.columns.get_loc(s)] = need[s][o]
mom = trailing_return(close, 28)
mom_same = mom.where(kron.notna())
comb = rank_blend(mom, kron)
SCORES = {"BASE": mom_same, "COMB": comb, "KRON": kron}


def weights(row):
    return xs_weights(row, MIN_ASSETS)


def run(name, cost):
    return run_portfolio(open_, SCORES[name], weights, decisions, STEP, cost)


def ci(x):
    return stationary_bootstrap_ci(x, MEAN_BLOCK, N_BOOT, 0, LEVEL)


def reading(lo, hi):
    return "Passed" if lo > 0 else "Failed" if hi < 0 else "Inconclusive"


rows = []
res = {(n, c): run(n, c) for n in SCORES for c in (0, 5, 10, 20)}
for name in SCORES:
    r, _ = res[(name, PRIMARY_COST)]
    lo, hi = ci(r["net"].to_numpy())
    st = period_stats(r["net"].to_numpy(), PPY)
    rows.append({"kind": "arm", "name": name, "cost_bps": PRIMARY_COST, "n_periods": len(r), "mean_bps": st["mean_bps"], "ci_lo99_bps": lo * 1e4, "ci_hi99_bps": hi * 1e4,
                 "status": reading(lo, hi), "sharpe": st["sharpe"], "max_drawdown": st["max_drawdown"], "turnover_per_period": float(r["turnover"].mean()),
                 "gross_mean_bps": float(r["gross"].mean() * 1e4)})
for cost in (0, 5, 10, 20):
    diff = res[("COMB", cost)][0]["net"].to_numpy() - res[("BASE", cost)][0]["net"].to_numpy()
    lo, hi = ci(diff)
    rows.append({"kind": "paired_diff", "name": "COMB-BASE", "cost_bps": cost, "n_periods": len(diff), "mean_bps": diff.mean() * 1e4, "ci_lo99_bps": lo * 1e4, "ci_hi99_bps": hi * 1e4,
                 "status": reading(lo, hi), "primary": cost == PRIMARY_COST})
for name in ("COMB", "BASE"):  # flat between evaluated periods: entry and exit are both paid
    r, w = res[(name, PRIMARY_COST)]
    res[(name, "rt")] = r["gross"].to_numpy() - 2 * PRIMARY_COST / 1e4 * w.abs().sum(axis=1).to_numpy()
diff = res[("COMB", "rt")] - res[("BASE", "rt")]
lo, hi = ci(diff)
rows.append({"kind": "paired_diff_roundtrip", "name": "COMB-BASE", "cost_bps": PRIMARY_COST, "n_periods": len(diff), "mean_bps": diff.mean() * 1e4, "ci_lo99_bps": lo * 1e4, "ci_hi99_bps": hi * 1e4,
             "status": reading(lo, hi)})
ret = (open_.shift(-(1 + STEP)).to_numpy() / open_.shift(-1).to_numpy() - 1.0)  # realised open d+1 -> open d+1+step, by decision day d
dec_idx = np.array(decisions)
actual = ret[dec_idx]


def ic_series(score):
    out = []
    for p, a in zip(score.to_numpy()[dec_idx], actual):
        ok = ~np.isnan(p) & ~np.isnan(a)
        out.append(rank_ic_by_date(p[ok][None, :], a[ok][None, :])[0] if ok.sum() >= MIN_ASSETS else np.nan)
    return np.array(out)


ics = {n: ic_series(SCORES[n]) for n in SCORES}
for n, x in ics.items():
    x2 = x[~np.isnan(x)]
    lo, hi = ci(x2)
    rows.append({"kind": "rank_ic", "name": n, "n_periods": len(x2), "mean": x2.mean(), "ci_lo99": lo, "ci_hi99": hi})
ok = ~np.isnan(ics["COMB"]) & ~np.isnan(ics["BASE"])
d_ic = ics["COMB"][ok] - ics["BASE"][ok]
lo, hi = ci(d_ic)
rows.append({"kind": "rank_ic_diff", "name": "COMB-BASE", "n_periods": int(ok.sum()), "mean": d_ic.mean(), "ci_lo99": lo, "ci_hi99": hi})
corr = []
for p, m in zip(kron.to_numpy()[dec_idx], mom_same.to_numpy()[dec_idx]):
    k = ~np.isnan(p) & ~np.isnan(m)
    if k.sum() >= MIN_ASSETS:
        corr.append(rank_ic_by_date(p[k][None, :], m[k][None, :])[0])
rows.append({"kind": "kronos_vs_momentum_rank_corr", "name": "mean", "n_periods": len(corr), "mean": float(np.nanmean(corr))})
r_c, r_b = res[("COMB", PRIMARY_COST)][0], res[("BASE", PRIMARY_COST)][0]
yearly = pd.DataFrame({"year": r_c["start"].dt.year, "diff_bps": (r_c["net"] - r_b["net"]) * 1e4}).groupby("year")["diff_bps"].agg(["mean", "count"])
for yr, v in yearly.iterrows():
    rows.append({"kind": "paired_diff_by_year", "name": str(yr), "n_periods": int(v["count"]), "mean_bps": v["mean"]})

table = pd.DataFrame(rows)
table.to_csv(ROOT / "outputs" / "stage13b_incremental.csv", index=False)
np.savez(ROOT / "outputs" / "stage13b_matrices.npz", kron=kron.to_numpy()[dec_idx], mom=mom_same.to_numpy()[dec_idx], actual=actual, decisions=dec_idx, assets=UNIVERSE)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print("\n", table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
prim = table[(table["kind"] == "paired_diff") & (table["cost_bps"] == PRIMARY_COST)].iloc[0]
print(f"\nPRIMARY (COMB - BASE, {PRIMARY_COST} bps, {int(prim['n_periods'])} periods): {prim['mean_bps']:+.1f} bps per period, 99% CI [{prim['ci_lo99_bps']:+.1f}, {prim['ci_hi99_bps']:+.1f}] -> {prim['status']}")
print(f"half-width {(prim['ci_hi99_bps'] - prim['ci_lo99_bps']) / 2:.1f} bps (a true gain smaller than this could not have passed). A PASS would only be an upper bound (pretraining contamination).")
