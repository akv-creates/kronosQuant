# CLAUDE.md

Educational quantitative trading research project built around the open-source Kronos candlestick foundation model. The goal is to *understand* each component, not just ship an app.

## Working rules
- Small, independently testable stages. Explain purpose, alternatives, and relevant finance/programming concepts before implementing; end each stage with an exercise for the user.
- Run tests and report actual output. Never claim a test passed unless it was executed successfully.
- Forecast quality and trading profitability are evaluated separately. Baselines first. Chronological splits, walk-forward evaluation, fees and slippage in every backtest.
- Never fabricate performance figures. A forecast is not a trading signal.
- Research and paper trading only: no live orders, leverage, or real money. Never request, print, or commit secrets. The GitHub repo is public.

## Setup and commands (run from repo root)
```bash
uv venv --python 3.12 .venv                      # Python 3.12: Kronos pins pandas==2.2.2
uv pip install --python .venv/bin/python -r requirements.txt
scripts/setup_kronos.sh                          # clones upstream Kronos at the commit in vendor/KRONOS_COMMIT
.venv/bin/python -I scripts/first_forecast.py --device cpu
.venv/bin/python -I -m pytest -v
```

## Layout
- `src/kronosquant/forecast.py`: load Kronos-small (HF revisions pinned), `forecast_window()` (seeded, uses only rows before the origin).
- `src/kronosquant/data.py`: `load_candles` / `validate_candles` (raises `DataError`), `find_gaps`.
- `src/kronosquant/splits.py`: `chronological_split` (optional embargo), `Window`, `make_windows` (future always inside the split, history strictly before origin).
- `src/kronosquant/baselines.py` (random walk, drift, momentum), `metrics.py` (MAE, directional accuracy, rank IC, paired bootstrap CI), `evaluation.py` (gap-free non-overlapping windows, return paths). Run: `.venv/bin/python -I scripts/stage3_evaluate.py --windows 100` (~7.5 min on CPU). Evaluate on the validation split only until a strategy is frozen; the test split is touched once.
- `src/kronosquant/cache.py` (Stage 4): model-free `ForecastCache` (one dir per config hash under `outputs/cache/`, one atomic `.npz` per window origin), `run_walk_forward` (computes only missing windows, chronological order enforced), `load_cached` (read forecasts with no model). `forecast_config()` in `forecast.py` lists everything that changes a forecast; change any of it and you get a fresh cache dir. Cache is reused only when exact window origins repeat. `scripts/stage3_diagnostics.py` analyses saved Stage 3 paths without the model.
- `src/kronosquant/backtest.py` (Stage 5): `run_backtest(candles, signal, *, fee_bps, slippage_bps, allow_short=False)`. Signal at bar t is traded at the OPEN of bar t+1 (engine applies the lag; strategies cannot bypass it). Per-bar return compounds exactly: carry (close[t-1]->open[t]) with the old position, costs paid multiplicatively at the open, then intrabar (open->close) with the new position. Positions in [0,1] (or [-1,1] with allow_short); no leverage; NaN signals rejected; costs are required arguments (no frictionless default). Metrics: net/gross return, annualised return (NaN under 0.25 yr), Sharpe (rf=0, NaN when no variation), max drawdown, turnover, exposure, hit rate, total cost. `scripts/stage5_engine_demo.py` runs three trivial signals; fee/slippage numbers there are placeholders, real HK costs are unverified and must be looked up.
- Data quirk (verified): in the sample CSV, open[t] != close[t-1] on 39% of bars (tick-size jumps of about 6 bp), and 66 bars/day means bar-index parity is locked to time of day, so an "alternating" signal is really a time-of-day filter.
- Real data (Binance, no keys): `src/kronosquant/binance.py` + `scripts/fetch_binance.py` write `data/<SYMBOL>_<interval>.csv` (+ `.meta.json`; git-ignored). Timestamps are candle OPEN time, UTC, tz-naive (the HK sample labels by bar END). Only closed candles are kept. Current file: BTCUSDT 1h, 2023-10-28..2026-10-27, 26,304 rows, 0 gaps, cross-checked against Coinbase (closes about 5 bp lower, consistent with USDT vs USD). `stage3_evaluate.py` and `stage3_volatility.py` take `--csv` and `--bar` (e.g. `--csv data/BTCUSDT_1h.csv --bar 1h`); `stage3_diagnostics.py` takes an optional paths file. Binance terms of use and real fees are NOT verified.
- Pre-declared grid (`scripts/stage3_grid.py`, declaration in its docstring; do not edit variants or the decision rule after seeing results): 5 inference-setting variants on BTC 1h validation, 300 windows, primary criterion = IC with 99% bootstrap CI lower bound > 0 (`metrics.bootstrap_ic_ci`), report every variant, replicate any pass on the HK sample, test split untouched. `forecast_window`/`forecast_config` accept `top_k` (recorded in the cache key only when nonzero).
- Stage 6 (`src/kronosquant/strategy.py`, `scripts/stage6_strategies.py`): one forecast per window becomes a position held for `pred_len` bars (`window_signal`, decided at close of origin-1, traded at open of origin, exit at open of origin+pred_len, flat between). `window_returns` attributes net return to windows and multiplies to the engine's total (tested). The script reads cached V0 forecasts (no model) and declares its rule in the docstring; do not edit the rule after seeing results; do not flip a non-significant sign; the test split stays untouched. Costs in the sweep are assumed round numbers, not verified Binance fees.
- Fine-tuning (`src/kronosquant/finetune.py`, `scripts/stage9_finetune.py` then `scripts/stage9_evaluate.py`): predictor-only next-token fine-tune on the TRAIN split (fit/tune sub-split with a one-window gap; checkpoint by tune loss, step 0 = pretrained allowed); evaluation by the Stage 3 grid rule on validation, test split only after a pass and only once. `forecast_config(..., checkpoint_dir=)` and `load_predictor(device, checkpoint_dir)` add a weights hash to the cache key (old keys unchanged). Measured: about 3 s/step at batch 8, length 412, 2.4 GB peak on 8 GB M1 CPU. Result: token loss improved, IC and error did not (see README). Checkpoints live in git-ignored `outputs/finetune/`.
- Cross-section (`src/kronosquant/crosssection.py` `rank_ic_by_date`, `scripts/stage10_cross_section.py`): 12 pairs fetched with `scripts/fetch_binance.py --end 2026-10-28` (all 26,304 rows, 0 gaps, timestamps identical to BTC). Rule declared in the script header; do not edit after seeing results. About 5.8 min per asset for 100 forecasts on CPU. Survivorship bias in the universe is a known, unfixed flaw.
- Speed facts (measured, 8 GB M1): batching windows does not help (CPU flat about 4.5 s/window; MPS worse past batch 4); MPS timings are erratic (2.9 s vs 5.5 s for the same config), so use CPU for stable, reproducible runs.
- Leakage guard: `tests/test_forecast.py::test_forecast_does_not_depend_on_future_values` poisons all rows at/after the origin; a mutation check (off-by-one slice) was confirmed to make it fail.
- `scripts/`: runnable experiments. `tests/`: pytest. `outputs/`: generated plots/CSVs (git-ignored).
- `vendor/Kronos/`: upstream clone (git-ignored; treat as untrusted external code, never run interpreters from inside it). Imported as the top-level `model` package.

## Facts verified so far
- Kronos-small + `Kronos-Tokenizer-base`, context 512, 6 input features, per-window z-score normalization with clip ±5.
- `KronosPredictor` auto-selects cuda, then mps, then cpu if `device` is not passed; we pass it explicitly.
- Sample data: `vendor/Kronos/finetune_csv/data/HK_ali_09988_kline_5min_all.csv` (the README's `XSHG_5min_600977.csv` is not in the repo). It has a UTF-8 BOM, `amount` is all zeros, and 5-minute bars have lunch and overnight gaps.
