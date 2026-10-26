# kronosQuant

Educational research project: forecasting market candles with the open-source [Kronos](https://github.com/shiyu-coder/Kronos) foundation model, then testing honestly whether any forecast skill survives baselines, costs, and out-of-sample evaluation.

**Research and paper trading only. Nothing here is financial advice, and a forecast is not a trading signal.**

## Status
- Stage 0 (environment, layout): done.
- Stage 1 (first forecast): script runs on CPU and MPS.
- Stage 2 (data validation, chronological splits, leakage tests): done. 26 tests pass.
- Later stages: data layer, baselines, evaluation, walk-forward harness, backtester, strategy research, paper trading, Streamlit dashboard.

## Quick start
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
scripts/setup_kronos.sh
.venv/bin/python -I scripts/first_forecast.py --device cpu
```

## Caveat for Stage 1
One forecast from one window proves nothing about skill. Kronos may also have seen this stock during pretraining, so even a good-looking result on the sample data would not be out-of-sample evidence.
