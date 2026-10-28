# kronosQuant

Educational research project: forecasting market candles with the open-source [Kronos](https://github.com/shiyu-coder/Kronos) foundation model, then testing honestly whether any forecast skill survives baselines, costs, and out-of-sample evaluation.

**Research and paper trading only. Nothing here is financial advice, and a forecast is not a trading signal.**

## Status
- Stage 0 (environment, layout): done.
- Stage 1 (first forecast): script runs on CPU and MPS.
- Stage 2 (data validation, chronological splits, leakage tests): done.
- Stage 3 (forecast quality vs baselines, validation split only): done. First result, one stock (HK 09988), 100 non-overlapping 1-hour windows, Kronos-small zero-shot with default sampling: Kronos did **not** beat the random-walk baseline on error (its extra error was statistically distinguishable from zero), and its directional accuracy (49.0%, n=96) matched the base rate (49.0% of windows rose). This is a narrow result, not a verdict on Kronos. The test split is untouched. Diagnostics (`scripts/stage3_diagnostics.py`): shrinking Kronos toward zero never helped, and its end-of-hour forecast correlated -0.51 with the previous hour's move (it mostly predicts reversals), which this stock did not show.
- Real data: Binance BTCUSDT 1h (3 years, no gaps) integrated. Same Stage 3 test, 100 validation windows of 12 hours: Kronos-small again did **not** beat the random walk on return error (its excess error was distinguishable from zero), direction 54% (n=100) versus a 57% base rate of up-moves, and its end-of-window forecast again correlated about -0.53 with the previous move (reversal tilt) that the data did not show. Volatility check (cached forecasts, predicted vs actual candle range): on Hong Kong, Kronos beat flat baselines but a same-time-of-day baseline beat Kronos (point estimate; its interval just touched zero); on BTC all baselines and Kronos were statistically indistinguishable. This volatility check was chosen after seeing the return results, so treat it as exploratory. Two assets, one model size, default sampling: not a general verdict.
- Pre-declared inference grid (`scripts/stage3_grid.py`, BTC 1h, validation split, 300 non-overlapping windows per variant, CPU, primary criterion: lower end of the 99% bootstrap interval for IC above zero). **No variant passed.** IC (99% interval): baseline -0.062 [-0.225, +0.096]; greedy (top_k=1) -0.022 [-0.179, +0.135]; T=0.5 -0.068 [-0.230, +0.090]; lookback 128 +0.062 [-0.096, +0.202]; horizon 4 bars +0.098 [-0.076, +0.261]. Direction accuracy was 47.7% to 53.0% against base rates of 49.0% (52.7% for the 4-bar horizon). In all five variants Kronos had higher path error than the random walk, and the 99% interval for the excess error excluded zero (excess +0.0001 to +0.0013 in mean absolute log-return). The forecast stayed anti-correlated with the previous move (corr -0.37 to -0.52) in every variant. The best observed IC (+0.098) is about 1.7 standard errors from zero (standard error about 0.058 at n=300), the kind of maximum you expect from five noisy draws. Nothing was replicated on the HK sample because nothing passed; the test split is untouched. Scope: one asset, one model size, 5 settings; this does not rule out gains from fine-tuning, a larger model, or other assets.
- Stage 5 (backtest engine): done. 59 tests pass; the look-ahead (execution lag) rule was mutation-checked. The engine is tested, not any strategy: the demo signals are trivial and the cost numbers are placeholders.
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
