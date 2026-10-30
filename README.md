# KronosQuant

An open research project that asks one question honestly: **can the open-source [Kronos](https://github.com/shiyu-coder/Kronos) candlestick model forecast markets well enough to be useful, once you test it against simple baselines, trading costs and data it has not seen?**

KronosQuant is the test bench for answering that. It downloads public market data, produces and caches Kronos forecasts, scores them against simple rules, simulates trading with realistic costs, and shows every result in a dashboard, including the failures.

**Live dashboard:** <https://kronosquant-kqnmvjqt5zxhytdtndqt3z.streamlit.app/> (a read-only view of the saved results; nothing is computed when you open it)

> **Research and paper trading only.** Nothing here places real orders or uses real money, and nothing is investment advice. A forecast is not a trading signal.

## What the research has found so far

Nineteen comparisons were declared before they were run, and the results were read against the declared rule. In short:

| Question | Result |
|---|---|
| Does Kronos-small forecast BTC better than "the price stays put"? | **No.** Its error was higher than the random walk in every setting tried, and the difference was statistically clear. Its rank correlation with outcomes (IC) ranged from -0.07 to +0.10, always with an interval that included zero. |
| Does fine-tuning, a different temperature, or averaging more samples help? | **No.** Fine-tuning improved the model's fit to the data (its loss) but not its forecasts. 20 sampled paths instead of 5 changed nothing measurable. |
| Can it rank 12 crypto assets better than chance? | **Not shown.** IC -0.017, 99% interval [-0.101, +0.076]. |
| Can it forecast volatility better than a one-line rule? | **Not shown.** It did not reliably beat any of three simple baselines, and a one-line "recent volatility" rule tracked volatility at least as well. |
| Does trading its forecasts make money after costs? | **No, in this period.** Kronos long/short lost 47.7% net at 10 bps in a market that fell 41.6%. |
| Do simple momentum rules work on daily bars? | **One marginal pass.** Cross-sectional momentum earned +46.7 bps per 5 days, but the lower end of its interval is about +1 bp, almost all of the gain came before 2022, and at 20 bps its interval includes zero. |
| Does Kronos add anything on top of that momentum rule? | **Inconclusive.** +18.8 bps per period, 99% interval [-32.7, +73.3]. |

**Bottom line:** there is no evidence here that Kronos has a usable trading edge. That is not the same as proof that it has none: most of the tests were too small to detect a modest effect, so most results are "inconclusive" rather than "reliably wrong". The full numbers and caveats are in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

## How it works

```
Binance public candles ──> validated data ──> chronological split (train / validation / sealed test)
                                                        │
              Kronos forecasts (cached, seeded) ────────┤
              simple baselines (random walk, momentum) ─┤
                                                        ▼
                 forecast quality (IC, error, direction)  ──>  trading simulation (fees, slippage)
                                                        ▼
                          paper account with risk limits  ──>  dashboard
```

The rules that keep the results honest:

- **Baselines first.** Every model result is shown next to a simple rule it has to beat.
- **No peeking.** Splits are chronological. Tests poison future data to prove no forecast or signal can see it.
- **Rules before results.** Each experiment script states its pass/fail rule in its header before it runs, and the header is not edited afterwards.
- **Forecast quality and profit are judged separately.** A forecast can be good and unprofitable, or the reverse.
- **Costs always.** Every backtest charges fees. The cost figures are round-number assumptions, not verified exchange fees.
- **The test split stays sealed** until something passes on validation, and then it is opened once.
- **Failures stay in.** Nothing is removed or re-run to look better.

## Quick start

You need Python 3.12 (managed with [uv](https://docs.astral.sh/uv/)), several GB of disk (PyTorch is the largest part), and no API keys.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-research.txt
scripts/setup_kronos.sh
```

Run the tests (about 40 seconds):

```bash
.venv/bin/python -I -m pytest
```

Download market data (public, no keys; `data/` is not committed) and make a first forecast on the CPU:

```bash
.venv/bin/python -I scripts/fetch_binance.py --symbol BTCUSDT --interval 1h --years 3 --end 2026-10-28
.venv/bin/python -I scripts/first_forecast.py --device cpu
```

Open the dashboard:

```bash
.venv/bin/python -I -m streamlit run dashboard/app.py
```

Result files live in `outputs/` and are not committed. A fresh clone shows the **published snapshot** in `published/` (see below) until you run the experiments yourself, after which your own `outputs/` take over. Each experiment's exact command is on the dashboard's **Reports** page. Forecasts are cached, so rerunning an experiment reproduces its saved numbers, and long runs resume where they stopped.

## The hosted dashboard

The dashboard can be hosted for anyone to open without installing anything (see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for the Streamlit Community Cloud steps). The hosted version is built to be safe and cheap to run:

- It reads a small snapshot of saved results from `published/`, so visitors never trigger model inference, backtests or downloads. `scripts/publish_results.py` rebuilds the snapshot, and each file's SHA-256 is recorded in `published/manifest.json`.
- It needs only `requirements.txt` (Streamlit, Plotly, pandas, numpy), not PyTorch.
- Raw exchange candles are not published, because Binance's data terms restrict redistribution. The hosted Strategies page therefore shows precomputed paper-account runs; the interactive version runs on a local install with the data downloaded.
- It holds no secrets and needs none.

## The dashboard

| Page | What it shows |
|---|---|
| Home | A summary of the evidence, headline counts, strategy curves and every primary comparison with its 99% interval |
| Experiments | Every experiment, its outcome under the declared rule, and the caveats that go with a pass |
| Strategies | A paper account you can configure (fees, position limits, kill switch) and run on placeholder strategies |
| Models | Kronos-small variants against the random walk, with pinned model revisions |
| Datasets | Downloaded data, row counts, gaps, fingerprints and the chronological split |
| Reports | How to reproduce each experiment, its declared rule, downloadable results and a glossary |

The dashboard only reads result files and runs the paper account on the validation split. It never loads the test split, runs no model and trades nothing.

## Repository layout

```
src/kronosquant/     the library
  config.py            shared settings: asset universe, Kronos default settings, confidence level
  data.py, splits.py   validated candles; chronological splits and windows
  forecast.py, cache.py  Kronos inference (pinned revisions) and the resumable forecast cache
  baselines.py, metrics.py, evaluation.py, crosssection.py, volatility.py   scoring and statistics
  backtest.py, strategy.py, portfolio.py, paper.py   trading simulations and the risk-limited paper account
  finetune.py          optional fine-tuning loop
  experiments.py, dashboard_data.py, glossary.py   what the dashboard reads
scripts/             one runnable script per experiment (rule declared in each header)
dashboard/           Streamlit app: app.py, views/ (one file per page), common.py, theme.py, charts.py
tests/               pytest suite
docs/                EXPERIMENT_LOG.md (all results), DEPLOYMENT.md (hosting), PRODUCT_CONTEXT.md (product thinking)
published/           committed snapshot of results for the hosted dashboard (rebuilt by scripts/publish_results.py)
vendor/              upstream Kronos is cloned here by setup_kronos.sh (not committed)
```

## Limitations

- **Small samples.** Many tests have too few windows to detect a modest effect, which is why most results read "inconclusive". The interval half-widths are printed with each result.
- **Costs are assumed.** 10 bps one-way is a round number. Real Binance fees, slippage and the cost of shorting are not modelled or verified.
- **Survivorship bias.** The 12-asset universe was picked from assets that exist today.
- **One exchange, one model size, a few horizons.** Results do not generalise to other markets, to Kronos-base, or to other settings.
- **Possible pretraining overlap.** Kronos may have seen parts of the historical data it is tested on, so even a good result on that period would be an upper bound. Only a failure is decisive there.
- **Hardware.** All runs used CPU on an 8 GB laptop, which limits model size and the number of forecasts.
- **Not a product yet.** There is no live data feed and no broker integration, by design.

## More documentation

- [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md): every experiment, rule, number and caveat.
- [docs/PRODUCT_CONTEXT.md](docs/PRODUCT_CONTEXT.md): market research, viability and where this could go.
- [CLAUDE.md](CLAUDE.md): working rules and technical notes for contributors and coding assistants.

## Disclaimer

KronosQuant is an experimental quantitative research and paper-trading project. It is not a live trading platform, brokerage or advisory service, and nothing in this repository or on its dashboard is financial, investment, legal or tax advice or a recommendation to buy or sell anything. Every result comes from historical data and simulated trades with assumed costs, does not predict future results, and no result here has been shown to be a reliable, tradable edge. Use it at your own risk.

The upstream Kronos code and models are MIT-licensed by their authors. This repository does not yet include a licence file of its own.
