"""Stage 7 demo: run two trivial strategies through the paper account, with and without risk limits.

This shows how the account and its limits behave. It is NOT a strategy result: the strategies are placeholders, the limits
and costs are round numbers chosen before looking (real Binance costs are unverified), and it runs on the validation split only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from kronosquant.data import load_candles  # noqa: E402
from kronosquant.forecast import ROOT  # noqa: E402
from kronosquant.paper import RiskLimits, run_paper  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402

candles = load_candles(ROOT / "data" / "BTCUSDT_1h.csv")
val = chronological_split(len(candles))["val"]
df = candles.iloc[val.start : val.stop].reset_index(drop=True)
print(f"BTC 1h validation: {len(df)} bars, {df['timestamps'].iloc[0]} .. {df['timestamps'].iloc[-1]}")

always_long = lambda history, position: 1.0
momentum = lambda history, position: float(len(history) > 12 and history["close"].iloc[-1] > history["close"].iloc[-13])

NO_BINDING = RiskLimits(max_position=1.0, max_order=2.0, max_daily_loss=0.99, max_drawdown=0.99)  # effectively off, but explicit
SAFE = RiskLimits(max_position=0.5, max_order=0.25, max_daily_loss=0.03, max_drawdown=0.15)

rows = []
for name, strat in [("always_long", always_long), ("momentum_12", momentum)]:
    for lname, limits in [("no_limits", NO_BINDING), ("safe_limits", SAFE)]:
        r = run_paper(df, strat, limits=limits, fee_bps=10, slippage_bps=0)
        rows.append({"strategy": name, "limits": lname, **{k: r.metrics[k] for k in ["net_return", "max_drawdown", "sharpe", "turnover_total", "exposure", "kill_switch", "bars_in_daily_halt", "n_events"]}})
        if lname == "safe_limits":
            print(f"\n{name} with safe limits, events by rule:\n{r.events['rule'].value_counts().to_string() if len(r.events) else '  none'}")
            if r.metrics["kill_switch"]:
                tripped = r.events[r.events["rule"] == "kill_switch_tripped"].iloc[0]
                print(f"  kill switch tripped at {tripped['timestamp']}; flat for the remaining {len(df) - int(tripped['bar']) - 1} bars")

pd.set_option("display.width", 220)
print("\n", pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print("\nCosts: 10 bps one-way. Placeholder strategies; this demonstrates the mechanics, not a finding.")
