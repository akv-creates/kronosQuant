"""Export per-bar equity curves for the dashboard (BTC 1h validation, 10 bps one-way): Kronos long/short, momentum long/short, buy-and-hold.

Same windows, signals and engine as scripts/stage6_strategies.py (cached V0 forecasts, no model run). It adds no new test: it only saves the
curves, and it CHECKS itself against outputs/stage6_strategies.csv so the dashboard cannot drift from the declared Stage 6 results.
Writes outputs/equity_curves.csv (per bar) and outputs/kronos_window_returns.csv (per window, net of costs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from kronosquant.backtest import run_backtest  # noqa: E402
from kronosquant.baselines import BASELINES  # noqa: E402
from kronosquant.cache import ForecastCache, load_cached  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.evaluation import baseline_paths, paths_from_forecasts, select_windows  # noqa: E402
from kronosquant.config import KRONOS_V0 as V0, ROOT  # noqa: E402
from kronosquant.forecast import forecast_config  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402
from kronosquant.strategy import directions_from_forecast, window_returns, window_signal  # noqa: E402

COST = 10
H = V0["pred_len"]

candles = load_candles(ROOT / "data" / "BTCUSDT_1h.csv")
split = chronological_split(len(candles))["val"]
windows = select_windows(candles, split, V0["lookback"], H, 300, "1h")
cache = ForecastCache(ROOT / "outputs" / "cache", forecast_config(candles, "cpu", V0["lookback"], H, V0["T"], V0["top_p"], V0["sample_count"], V0["top_k"]))
forecasts = load_cached(windows, cache)
pred_end = paths_from_forecasts(candles, windows, forecasts)[:, -1]
mom_end = baseline_paths(candles, windows, BASELINES["momentum"])[:, -1]
val = candles.iloc[split.start : split.stop].reset_index(drop=True)
keep = [i for i, w in enumerate(windows) if 1 <= w.origin - split.start and w.origin - split.start + H < len(val)]
origins = [windows[i].origin - split.start for i in keep]
pred_end, mom_end = pred_end[keep], mom_end[keep]

curves, results = {}, {}
for name, directions in {"kronos_longshort": directions_from_forecast(pred_end, 0.0, True), "momentum_longshort": directions_from_forecast(mom_end, 0.0, True)}.items():
    r = run_backtest(val, window_signal(len(val), origins, directions, H), fee_bps=COST, slippage_bps=0, allow_short=True)
    curves[name], results[name] = r.frame["equity"].to_numpy(), r
bh = run_backtest(val, pd.Series(1.0, index=val.index), fee_bps=COST, slippage_bps=0)
curves["buy_hold"] = bh.frame["equity"].to_numpy()

# Self-check against the declared Stage 6 results
ref = pd.read_csv(ROOT / "outputs" / "stage6_strategies.csv")
for name, net in [("kronos_longshort", results["kronos_longshort"].metrics["net_return"]), ("momentum_longshort", results["momentum_longshort"].metrics["net_return"]),
                  ("buy_hold_whole_split", bh.metrics["net_return"])]:
    expected = float(ref[(ref["strategy"] == name) & (ref["cost_bps"] == COST)]["net_return"].iloc[0])
    assert abs(net - expected) < 1e-9, f"{name}: recomputed {net} != saved {expected}"
    print(f"{name}: net {net:+.4f} matches Stage 6")

out = ROOT / "outputs"
pd.DataFrame({"timestamp": val["timestamps"], **curves}).to_csv(out / "equity_curves.csv", index=False)
w = window_returns(results["kronos_longshort"].frame["net_ret"].to_numpy(), origins, H)
pd.DataFrame({"window_end": val["timestamps"].iloc[[o + H for o in origins]].to_numpy(), "net_return": w}).to_csv(out / "kronos_window_returns.csv", index=False)
print(f"saved equity_curves.csv ({len(val)} bars) and kronos_window_returns.csv ({len(w)} windows)")
