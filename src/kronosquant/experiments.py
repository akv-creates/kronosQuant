"""The registry of experiments shown on the dashboard.

Each entry says what was run (script, dataset, settings), which result files it writes and how to reproduce it. A new experiment script
needs an entry here, a results CSV in `outputs/`, and (if it has a declared primary comparison) a row in
`dashboard_data.scorecard`. A `caveat` is shown next to any result that passed, so a pass is never displayed without its limits.
"""

EXPERIMENTS = [
    dict(id="exp_stage3_quality", name="Forecast quality vs baselines", stage="Stage 3", model="Kronos-small (pretrained)", dataset="HK 09988 5min sample; BTCUSDT 1h",
         script="scripts/stage3_evaluate.py", results=["stage3_table.csv", "stage3_BTCUSDT_1h_table.csv"], period="validation",
         config="lookback 400, 120/12 bars, T 1.0, top_p 0.9, 1 sample", seed="0", costs="none (forecast quality only)", baselines="random walk, drift, momentum",
         command=".venv/bin/python -I scripts/stage3_evaluate.py --windows 100 --csv data/BTCUSDT_1h.csv --bar 1h"),
    dict(id="exp_grid_btc1h", name="Inference grid (5 variants)", stage="Grid", model="Kronos-small (pretrained)", dataset="BTCUSDT 1h", script="scripts/stage3_grid.py",
         results=["grid_BTCUSDT_1h.csv"], period="validation", config="V0 lookback 400, 12 bars, T 1.0, top_p 0.9, 5 samples; V1-V4 change one setting each",
         seed="window origin", costs="none (forecast quality only)", baselines="random walk, momentum", command=".venv/bin/python -I scripts/stage3_grid.py"),
    dict(id="exp_stage6_window_strategy", name="Window strategy after costs", stage="Stage 6", model="Kronos-small (cached V0 forecasts)", dataset="BTCUSDT 1h",
         script="scripts/stage6_strategies.py", results=["stage6_strategies.csv"], period="validation", config="sign of forecast, hold 12 bars, flat between windows",
         seed="window origin", costs="10 bps one-way (assumed, unverified); sweep 0/5/10/20", baselines="momentum, contrarian, always long, random long/short null",
         command=".venv/bin/python -I scripts/stage6_strategies.py"),
    dict(id="exp_stage9_finetune", name="Fine-tuned Kronos", stage="Stage 9", model="Kronos-small fine-tuned on BTC train split", dataset="BTCUSDT 1h",
         script="scripts/stage9_evaluate.py", results=["stage9_evaluate_val.csv"], period="validation", config="1000 steps, lr 1e-5, batch 8, checkpoint by tune loss",
         seed="window origin (inference); 0 (training)", costs="none (forecast quality only)", baselines="pretrained Kronos, random walk",
         command=".venv/bin/python -I scripts/stage9_finetune.py && .venv/bin/python -I scripts/stage9_evaluate.py"),
    dict(id="exp_stage10_crosssection", name="Cross-sectional ranking", stage="Stage 10", model="Kronos-small (pretrained)", dataset="12 Binance USDT pairs, 1h",
         script="scripts/stage10_cross_section.py", results=["stage10_cross_section.csv"], period="validation", config="V0 settings, 100 dates, 12 assets",
         seed="window origin", costs="none (ranking quality only)", baselines="reversal, momentum, vol-scaled Kronos", command=".venv/bin/python -I scripts/stage10_cross_section.py"),
    dict(id="exp_stage11_volatility", name="Volatility forecast", stage="Stage 11", model="Kronos-small (cached V0 forecasts)", dataset="12 Binance USDT pairs, 1h",
         script="scripts/stage11_volatility.py", results=["stage11_primary.csv", "stage11_per_asset.csv"], period="validation", config="mean candle range over 12 bars, 11 non-BTC assets",
         seed="window origin", costs="none (forecast quality only)", baselines="history average, last 48 bars, same hour of day", command=".venv/bin/python -I scripts/stage11_volatility.py"),
    dict(id="exp_stage13_signal_screen", name="Daily signal screen (momentum, reversal)", stage="Stage 13a", model="none (classical signals)", dataset="12 Binance USDT pairs, 1d, 2019 to 2025-08",
         script="scripts/stage13_signal_screen.py", results=["stage13_screen.csv", "stage13_by_year.csv"], period="primary window 2019-01-01 to 2025-08-15",
         config="5-day rebalance; TSMOM 28d, XSMOM 28d, REV 5d; terciles; equal weights", seed="none (deterministic); null seed 0", costs="10 bps one-way (assumed, unverified); shorts free (optimistic)",
         baselines="equal-weight long-only benchmark, random-signal null", command=".venv/bin/python -I scripts/stage13_signal_screen.py",
         caveat="Marginal pass: the interval's lower end is about 1 bp. The edge is concentrated in 2019-2021 (about +100 bps per period) and is about +2 bps per period from 2022; it fails at 20 bps; the universe has survivorship bias; short borrow and funding costs are not modelled."),
    dict(id="exp_stage13b_incremental", name="Kronos on top of momentum", stage="Stage 13b", model="Kronos-small (pretrained, daily bars)", dataset="12 Binance USDT pairs, 1d, every second 5-day period 2019 to 2025-08",
         script="scripts/stage13b_incremental.py", results=["stage13b_incremental.csv"], period="primary window 2019-01-01 to 2025-08-15 (half schedule, 242 periods)",
         config="lookback 120, 5 bars ahead, 5 paths; rank blend of XSMOM_28 and the Kronos score; terciles", seed="window origin", costs="10 bps one-way (assumed, unverified); shorts free (optimistic)",
         baselines="XSMOM_28 on the same assets", command=".venv/bin/python -I scripts/stage13b_incremental.py",
         caveat="Kronos may have seen this period in pretraining, so even a pass would be only an upper bound. The baseline is weak on this half schedule and universe."),
    dict(id="exp_stage12_samples", name="Sampling noise (5 vs 20 paths)", stage="Stage 12", model="Kronos-small (pretrained)", dataset="BTCUSDT 1h, all 438 validation windows",
         script="scripts/stage12_samples.py", results=["stage12_samples.csv"], period="validation", config="V0 settings; sample_count 5 and 20",
         seed="window origin", costs="none (forecast quality only)", baselines="5-path forecast, random walk, momentum", command=".venv/bin/python -I scripts/stage12_samples.py"),
]
