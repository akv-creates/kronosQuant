# kronosQuant

Educational research project: forecasting market candles with the open-source [Kronos](https://github.com/shiyu-coder/Kronos) foundation model, then testing honestly whether any forecast skill survives baselines, costs, and out-of-sample evaluation.

**Research and paper trading only. Nothing here is financial advice, and a forecast is not a trading signal.**

## Status
- Stage 0 (environment, layout): done.
- Stage 1 (first forecast): script runs on CPU and MPS.
- Stage 2 (data validation, chronological splits, leakage tests): done.
- Stage 3 (forecast quality vs baselines, validation split only): done. First result, one stock (HK 09988), 100 non-overlapping 1-hour windows, Kronos-small zero-shot with default sampling: Kronos did **not** beat the random-walk baseline on error (its extra error was statistically distinguishable from zero), and its directional accuracy (49.0%, n=96) matched the base rate (49.0% of windows rose). This is a narrow result, not a verdict on Kronos. The test split is untouched. Diagnostics (`scripts/stage3_diagnostics.py`): shrinking Kronos toward zero never helped, and its end-of-hour forecast correlated -0.51 with the previous hour's move (it mostly predicts reversals), which this stock did not show.
- Stage 4 (resumable forecast cache): done. 44 tests pass. Re-running the same 100 CPU windows through the cache reproduced the earlier Stage 3 numbers bit-for-bit, and a second run computed 0 forecasts (about 3s total instead of about 7.5 minutes).
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
