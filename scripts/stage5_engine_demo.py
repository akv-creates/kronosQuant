"""Stage 5 demo: exercise the backtest engine with three trivial signals on the validation split.

These are NOT strategies. They exist to show what the engine reports and how costs bite.
Fee/slippage numbers are placeholders for illustration; real HK costs must be looked up and set explicitly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kronosquant.backtest import run_backtest  # noqa: E402
from kronosquant.data import load_candles  # noqa: E402
from kronosquant.forecast import SAMPLE_CSV  # noqa: E402
from kronosquant.splits import chronological_split  # noqa: E402

candles = load_candles(SAMPLE_CSV)
val = chronological_split(len(candles))["val"]
df = candles.iloc[val.start : val.stop].reset_index(drop=True)
print(f"validation: {len(df)} bars, {df['timestamps'].iloc[0]} .. {df['timestamps'].iloc[-1]}")

signals = {
    "flat": np.zeros(len(df)),
    "buy_and_hold": np.ones(len(df)),
    "flip_every_bar": (np.arange(len(df)) % 2).astype(float),  # 0,1,0,1,... maximal turnover, no view
}
rows = []
for name, s in signals.items():
    for fee, slip in [(0, 0), (5, 2)]:
        m = run_backtest(df, pd.Series(s, index=df.index), fee_bps=fee, slippage_bps=slip).metrics
        rows.append({"signal": name, "fee+slip bps": fee + slip, **{k: m[k] for k in ["net_return", "gross_return", "sharpe", "max_drawdown", "turnover_per_year", "exposure", "total_cost"]}})
pd.set_option("display.width", 220)
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
