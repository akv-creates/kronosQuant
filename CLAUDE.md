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
- Leakage guard: `tests/test_forecast.py::test_forecast_does_not_depend_on_future_values` poisons all rows at/after the origin; a mutation check (off-by-one slice) was confirmed to make it fail.
- `scripts/`: runnable experiments. `tests/`: pytest. `outputs/`: generated plots/CSVs (git-ignored).
- `vendor/Kronos/`: upstream clone (git-ignored; treat as untrusted external code, never run interpreters from inside it). Imported as the top-level `model` package.

## Facts verified so far
- Kronos-small + `Kronos-Tokenizer-base`, context 512, 6 input features, per-window z-score normalization with clip ±5.
- `KronosPredictor` auto-selects cuda, then mps, then cpu if `device` is not passed; we pass it explicitly.
- Sample data: `vendor/Kronos/finetune_csv/data/HK_ali_09988_kline_5min_all.csv` (the README's `XSHG_5min_600977.csv` is not in the repo). It has a UTF-8 BOM, `amount` is all zeros, and 5-minute bars have lunch and overnight gaps.
