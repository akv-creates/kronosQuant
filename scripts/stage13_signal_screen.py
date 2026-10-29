"""Stage 13a: do simple, literature-motivated signals make money after costs on daily crypto bars? (no model; validation periods only)

DECLARED BEFORE RUNNING (do not edit after seeing results):
  why:        Stages 3 to 12 found no usable Kronos edge. Before asking whether Kronos adds anything (Stage 13b), test whether a simple
              baseline has a measurable, cost-surviving edge at all. Evidence for crypto momentum is mixed and mostly from working papers.
  data:       daily bars, 12 Binance USDT pairs (BTC ETH BNB XRP ADA DOGE LTC TRX LINK BCH ETC XLM), history to 2026-10-27. The list is
              picked today, so survivorship bias flatters long exposure and is NOT fixed. The test period is cut off at load time:
              rows dated 2026-03-23 or later are never read in this script.
  schedule:   a decision every 5 days (first decision 2018-12-31, close of day), traded at the open of the next day, held 5 days.
              All candidates use the same schedule. An asset joins a signal only once it has a full lookback of history.
  candidates (primary; K = 3, nothing is tuned):
              TSMOM_28  per asset, long if the trailing 28-day return is positive else short, equal weights, gross exposure 1.
              XSMOM_28  rank assets by trailing 28-day return; long the top third, short the bottom third; dollar neutral, gross 1.
              REV_5     rank by trailing 5-day return; long the bottom third, short the top third; dollar neutral, gross 1.
              REV_5 was suggested by an exploratory result in Stage 6 on 2025-08 to 2026-03, which is why that period is excluded
              from the primary window below.
  primary window: periods that start on or after 2019-01-01 and end on or before 2025-08-15 (about 480 periods). Nothing is fitted
              in it, so it is out of sample for fixed rules; it is long enough for more power than the hourly tests.
  primary cost: 10 bps one-way, all-in, on every change in weight (assumed, NOT a verified Binance cost). Shorting is assumed free
              of borrow and funding costs, which is optimistic and not modelled.
  primary test: mean net return per 5-day period; 99% stationary-block-bootstrap interval (mean block 4 periods, 5000 resamples, seed 0;
              blocks keep volatility clustering and serial dependence). Level 99% because K = 3 candidates.
              PASS = lower end of the interval above 0. FAILED = upper end below 0. INCONCLUSIVE = the interval includes 0.
  secondary (reported, never used to rescue a failed primary): costs 0, 5 and 20 bps; lookbacks 14 and 56 (sensitivity, not tuning);
              long-only TSMOM_28; REV_3 and REV_10; an equal-weight long-only benchmark; per-year means; the recent window
              2025-08-15 to 2026-03-22 (already seen in Stage 6); a null of 300 random-signal portfolios (share at least as good).
  after:      a PASS that is also positive after 20 bps and positive in 2022-01-01 to 2025-08-15 (publication-decay check) justifies
              Stage 13b (does Kronos add anything on that baseline?) and forward paper trading; the real test period is opened once,
              only then. If nothing passes, the simple-signal family is not shown to work on this universe; an INCONCLUSIVE result
              means the test lacked the power to tell, which the printed half-width states.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kronosquant.data import load_candles  # noqa: E402
from kronosquant.forecast import ROOT  # noqa: E402
from kronosquant.metrics import stationary_bootstrap_ci  # noqa: E402
from kronosquant.portfolio import decision_indices, long_only_weights, period_stats, run_portfolio, trailing_return, tsmom_weights, xs_weights  # noqa: E402

UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "TRXUSDT", "LINKUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT"]
TEST_START = pd.Timestamp("2026-03-23")  # rows from here on are never loaded
FIRST_DECISION, STEP, LEVEL, MEAN_BLOCK, N_BOOT, N_NULL = "2018-12-31", 5, 0.99, 4.0, 5000, 300
PPY = 365.0 / STEP
PRIMARY_COST = 10
WINDOWS = {"primary": (pd.Timestamp("2019-01-01"), pd.Timestamp("2025-08-15")), "recent": (pd.Timestamp("2025-08-15"), TEST_START)}
SPEC = {  # name -> (lookback, kind, primary?)
    "TSMOM_28": (28, "tsmom", True), "XSMOM_28": (28, "xs", True), "REV_5": (5, "rev", True),
    "TSMOM_14": (14, "tsmom", False), "TSMOM_56": (56, "tsmom", False), "XSMOM_14": (14, "xs", False), "XSMOM_56": (56, "xs", False),
    "TSMOM_28_longonly": (28, "longonly", False), "REV_3": (3, "rev", False), "REV_10": (10, "rev", False),
}

frames = {}
for s in UNIVERSE:
    c = load_candles(ROOT / "data" / f"{s}_1d.csv")
    frames[s] = c[c["timestamps"] < TEST_START].set_index("timestamps")  # hard cut: the test period is never read
dates = frames["BTCUSDT"].index
open_ = pd.DataFrame({s: f["open"].reindex(dates) for s, f in frames.items()})
close = pd.DataFrame({s: f["close"].reindex(dates) for s, f in frames.items()})
print(f"{len(dates)} daily rows {dates[0]:%Y-%m-%d} .. {dates[-1]:%Y-%m-%d}; assets listed by 2019-01-01: {int(close.loc['2019-01-01'].notna().sum())}", flush=True)
decisions = decision_indices(dates, FIRST_DECISION, STEP)


def weight_fn(kind):
    return {"tsmom": tsmom_weights, "xs": xs_weights, "rev": lambda row: xs_weights(-row), "longonly": long_only_weights}[kind]


def in_window(res, name):
    lo, hi = WINDOWS[name]
    return res[(res["start"] >= lo) & (res["end"] <= hi)]


rows, by_year, results = [], [], {}
for name, (lookback, kind, primary) in SPEC.items():
    scores = trailing_return(close, lookback)
    for cost in (0, 5, 10, 20):
        res, _ = run_portfolio(open_, scores, weight_fn(kind), decisions, STEP, cost)
        results[(name, cost)] = res
        for wname in WINDOWS:
            part = in_window(res, wname)
            if len(part) < 20:
                continue
            net = part["net"].to_numpy()
            lo, hi = stationary_bootstrap_ci(net, MEAN_BLOCK, N_BOOT, 0, LEVEL)
            st = period_stats(net, PPY)
            status = "Passed" if lo > 0 else "Failed" if hi < 0 else "Inconclusive"
            rows.append({"candidate": name, "primary": primary, "window": wname, "cost_bps": cost, "n_periods": len(part), "mean_bps": st["mean_bps"], "ci_lo99_bps": lo * 1e4, "ci_hi99_bps": hi * 1e4,
                         "status": status, "sharpe": st["sharpe"], "total_return": st["total_return"], "max_drawdown": st["max_drawdown"],
                         "turnover_per_period": float(part["turnover"].mean()), "gross_mean_bps": float(part["gross"].mean() * 1e4)})

# equal-weight long-only benchmark over eligible assets (those with a 28-day history), same schedule and cost
elig = trailing_return(close, 28)
bench, _ = run_portfolio(open_, elig, lambda r: np.where(np.isnan(r), 0.0, 1.0 / max(1, int((~np.isnan(r)).sum()))), decisions, STEP, PRIMARY_COST)
for wname in WINDOWS:
    part = in_window(bench, wname)
    st = period_stats(part["net"].to_numpy(), PPY)
    rows.append({"candidate": "EW_buy_hold_benchmark", "primary": False, "window": wname, "cost_bps": PRIMARY_COST, "n_periods": len(part), "mean_bps": st["mean_bps"], "ci_lo99_bps": np.nan, "ci_hi99_bps": np.nan,
                 "status": "", "sharpe": st["sharpe"], "total_return": st["total_return"], "max_drawdown": st["max_drawdown"], "turnover_per_period": float(part["turnover"].mean()),
                 "gross_mean_bps": float(part["gross"].mean() * 1e4)})

# random-signal null for the primary candidates: same assets, schedule, cost and weight rule, scores replaced by noise
table = pd.DataFrame(rows)
rng = np.random.default_rng(0)
for name in [n for n, v in SPEC.items() if v[2]]:
    lookback, kind, _ = SPEC[name]
    mask = trailing_return(close, lookback).isna().to_numpy()
    obs = float(table[(table["candidate"] == name) & (table["window"] == "primary") & (table["cost_bps"] == PRIMARY_COST)]["mean_bps"].iloc[0])
    null_means = []
    for _ in range(N_NULL):
        noise = pd.DataFrame(np.where(mask, np.nan, rng.normal(size=mask.shape)), index=dates, columns=close.columns)
        res, _ = run_portfolio(open_, noise, weight_fn(kind), decisions, STEP, PRIMARY_COST)
        null_means.append(in_window(res, "primary")["net"].mean() * 1e4)
    table.loc[(table["candidate"] == name) & (table["window"] == "primary") & (table["cost_bps"] == PRIMARY_COST), "null_share_at_least_as_good"] = float((np.array(null_means) >= obs).mean())

out = ROOT / "outputs"
table.to_csv(out / "stage13_screen.csv", index=False)
for name in [n for n, v in SPEC.items() if v[2]]:
    part = in_window(results[(name, PRIMARY_COST)], "primary")
    g = part.groupby(part["start"].dt.year)["net"].agg(["mean", "count"])
    for yr, r in g.iterrows():
        by_year.append({"candidate": name, "year": int(yr), "mean_bps": r["mean"] * 1e4, "n_periods": int(r["count"])})
pd.DataFrame(by_year).to_csv(out / "stage13_by_year.csv", index=False)

pd.set_option("display.width", 250)
cols = ["candidate", "window", "cost_bps", "n_periods", "mean_bps", "ci_lo99_bps", "ci_hi99_bps", "status", "sharpe", "total_return", "max_drawdown", "turnover_per_period"]
prim = table[(table["primary"]) & (table["window"] == "primary") & (table["cost_bps"] == PRIMARY_COST)]
print("\nPRIMARY (10 bps, primary window):\n", prim[cols + ["gross_mean_bps", "null_share_at_least_as_good"]].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print("\nSTRESS and SECONDARY (primary window):\n", table[(table["window"] == "primary") & ~((table["primary"]) & (table["cost_bps"] == PRIMARY_COST))][cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print("\nRECENT window (already seen, 10 bps):\n", table[(table["window"] == "recent") & (table["cost_bps"] == PRIMARY_COST)][cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print("\nMean net bps per 5-day period by year (primary candidates, 10 bps):\n", pd.DataFrame(by_year).pivot(index="year", columns="candidate", values="mean_bps").round(1).to_string())
best = prim.loc[prim["mean_bps"].idxmax()]
print(f"\nhalf-width of the best candidate's 99% interval: {(best['ci_hi99_bps'] - best['ci_lo99_bps']) / 2:.1f} bps per 5-day period (a mean below this cannot pass)")
print(f"PRIMARY PASSES: {[r.candidate for r in prim.itertuples() if r.status == 'Passed'] or 'none'}")
