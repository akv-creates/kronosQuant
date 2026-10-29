"""Stage 11: does pretrained Kronos-small forecast how VOLATILE the next 12 hours will be better than simple baselines? (validation split only)

DECLARED BEFORE RUNNING (do not edit after seeing results):
  why:        direction was not forecastable (Stages 3 to 10). Volatility usually is, and the Kronos paper reports a gain there.
              That claim is the authors' own, on their benchmarks; this tests it on our data.
  data:       the 12 Binance 1h assets and 100 validation windows of Stage 10, and the cached V0 forecasts from that stage
              (lookback 400, 12 bars, T 1.0, top_p 0.9, sample_count 5). No model is run here.
  target:     realized range of a window = mean over its 12 future bars of (high - low) / close.
  forecast:   Kronos = mean over the 12 forecast bars of (high - low) / close from the cached mean path.
  baselines:  hist_all (mean range over the 400-bar history), hist_recent (last 48 bars), same_time_of_day (history mean
              range at each future bar's hour of day). All use rows before the origin only (leakage-tested).
  error:      |log(forecast / realized)| per window and asset, smaller is better.
  primary:    the 11 assets other than BTC. (BTC volatility was already examined in an exploratory check, so it is
              reported separately and is NOT part of the pass decision.) For each date average the error difference
              (Kronos minus baseline) over the 11 assets, then bootstrap over dates (99% interval, because there are three
              comparisons). PASS only if the upper end of the interval is below 0 for ALL THREE baselines.
              Averaging over assets first respects that volatility moves together across assets on the same date.
  secondary (reported, not used to rescue a failed primary): per-asset errors; rank IC over windows within each asset;
              mean signed log ratio (bias: does Kronos forecast systematically too high or too low?); BTC alone.
              No rescaling of Kronos to fix its bias is done: that would be fitted on validation data.
  after:      a PASS would justify a volatility-targeted position-size test on validation through the paper account,
              then ONE look at the test split. A fail ends the volatility idea for this model on this universe.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kronosquant.cache import ForecastCache, load_cached  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.evaluation import select_windows  # noqa: E402
from kronosquant.forecast import ROOT, forecast_config  # noqa: E402
from kronosquant.metrics import information_coefficient, paired_bootstrap_ci  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402
from kronosquant.volatility import candle_range, kronos_range, log_ratio_errors, range_baselines, realized_range  # noqa: E402

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "TRXUSDT", "LINKUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT"]
V0 = dict(lookback=400, pred_len=12, T=1.0, top_p=0.9, top_k=0, sample_count=5)
LEVEL, N_DATES, BASELINES = 0.99, 100, ["hist_all", "hist_recent", "same_time_of_day"]

data = {s: load_candles(ROOT / "data" / f"{s}_1h.csv") for s in UNIVERSE}
btc = data["BTCUSDT"]
windows = select_windows(btc, chronological_split(len(btc))["val"], V0["lookback"], V0["pred_len"], N_DATES, "1h")
print(f"{len(windows)} windows x {len(UNIVERSE)} assets, validation split only")

actual, preds = {}, {}
for s in UNIVERSE:
    c = data[s]
    cache = ForecastCache(ROOT / "outputs" / "cache", forecast_config(c, "cpu", V0["lookback"], V0["pred_len"], V0["T"], V0["top_p"], V0["sample_count"], V0["top_k"]))
    forecasts = load_cached(windows, cache)  # raises if Stage 10 did not cache these
    rng = candle_range(c)
    actual[s] = realized_range(rng, windows)
    preds[s] = {"kronos": kronos_range(forecasts, windows), **range_baselines(rng, c["timestamps"], windows)}

others = [s for s in UNIVERSE if s != "BTCUSDT"]
rows = []
for b in BASELINES:
    diff = np.mean([log_ratio_errors(preds[s]["kronos"], actual[s]) - log_ratio_errors(preds[s][b], actual[s]) for s in others], axis=0)  # per date
    lo, hi = paired_bootstrap_ci(diff, level=LEVEL)
    rows.append({"baseline": b, "mean_err_kronos": np.mean([log_ratio_errors(preds[s]["kronos"], actual[s]).mean() for s in others]),
                 "mean_err_baseline": np.mean([log_ratio_errors(preds[s][b], actual[s]).mean() for s in others]),
                 "kronos_minus_baseline": diff.mean(), "ci_lo99": lo, "ci_hi99": hi, "PASS": bool(hi < 0)})
table = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(f"\nPRIMARY: 11 non-BTC assets, error difference (Kronos minus baseline), {LEVEL:.0%} CI over dates; negative = Kronos better\n")
print(table.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
table.to_csv(ROOT / "outputs" / "stage11_primary.csv", index=False)

per_asset = []
for s in UNIVERSE:
    row = {"asset": s, "kronos_err": log_ratio_errors(preds[s]["kronos"], actual[s]).mean(), "kronos_bias": np.log(preds[s]["kronos"] / actual[s]).mean(),
           "kronos_rankIC": information_coefficient(preds[s]["kronos"], actual[s])}
    for b in BASELINES:
        row[f"{b}_err"] = log_ratio_errors(preds[s][b], actual[s]).mean()
    row["hist_recent_rankIC"] = information_coefficient(preds[s]["hist_recent"], actual[s])
    per_asset.append(row)
pa = pd.DataFrame(per_asset)
pa.to_csv(ROOT / "outputs" / "stage11_per_asset.csv", index=False)
print("\nSECONDARY per asset (BTC is the exploratory one; bias = mean log(forecast/realized), negative = forecasts too low):\n")
print(pa.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print(f"\nmean rank IC over assets: Kronos {pa['kronos_rankIC'].mean():+.3f}, hist_recent {pa['hist_recent_rankIC'].mean():+.3f}")
print(f"Kronos best of 4 on error in {sum(pa['kronos_err'] < pa[[f'{b}_err' for b in BASELINES]].min(axis=1))} of {len(pa)} assets")
print(f"\nPRIMARY (Kronos beats all three baselines, 99% CI upper < 0): {bool(table['PASS'].all())}")
