# kronosQuant: product context

Written 2026-10-30. This file collects everything known about the project's goal, evidence, market research, viability and plan, so that someone (or a new session) can pick it up without the chat history. It states what is measured, what is researched, and what is only a judgement. Nothing here is financial advice.

> The repo is public and this file is committed to it, so everything here is readable by anyone. Remove or trim sections (market research, viability, plans) if you do not want them public.

---

## 1. Goal and how it evolved

1. **Started as** an educational quantitative research project: understand the open-source [Kronos](https://github.com/shiyu-coder/Kronos) candlestick foundation model by building each component, and test honestly whether any forecast skill survives baselines, costs and out-of-sample evaluation.
2. **The owner's current goal**: make it *really useful*, not "just a piece of code"; give *unique insights* that other software does not; and the most important outcome for users is **profit**.
3. **The honest tension:** the evidence so far says Kronos-small does not give a usable trading edge on the data tested (section 4). So the product must either find an edge elsewhere (section 8) or be valuable for a different reason (verification, risk, credibility; section 6).

Hard rules that apply to everything (from the project's working rules):
- Research and **paper trading only**. No live orders, leverage or real money. Never request, print or commit secrets.
- Forecast quality and trading profitability are evaluated **separately**. Baselines first. Chronological splits. Fees and slippage in every backtest.
- **Never fabricate or "tune to pass" results.** Rules are declared *before* running and not edited afterwards. The test split is opened only after a validation pass, and only once. (The owner once asked whether the passed count could be raised to 11 or 12 by fixing the code; this was declined because it would mean changing rules or data after seeing results.)

---

## 2. What exists (built and tested)

Repository: `/Users/apple/Gen AI apps/kronosQuant`, GitHub `akv-creates/kronosQuant` (public). Python 3.12 venv (uv), pandas 2.2.2, torch 2.14.1, Apple M1 with 8 GB RAM; CPU is used for stable timings. Kronos-small (24.7M parameters, context 512) with pinned Hugging Face revisions and a pinned upstream commit.

| Stage | What | Status |
|---|---|---|
| 0-1 | Environment, layout, first Kronos forecast (CPU and MPS) | done |
| 2 | Data validation, chronological splits, leakage tests | done |
| 3 | Forecast quality vs baselines (HK stock sample, BTC 1h) | done |
| 4 | Resumable forecast cache keyed by config, data fingerprint, weights hash | done |
| 5 | Backtest engine (execution lag, costs required, no leverage) | done |
| grid | Pre-declared 5-variant inference grid on BTC 1h | done, no variant passed |
| 6 | Window strategies through the engine | done, primary failed |
| 7 | Bar-by-bar paper account with risk limits | done |
| 8 | Streamlit dashboard (restyled to the owner's design reference: navy sidebar, light cards, 6 pages) | done; redesign uncommitted |
| 9 | Fine-tuning Kronos-small on the BTC train split | done, primary failed |
| 10 | Cross-sectional ranking test, 12 assets | done, primary failed |
| 11 | Volatility forecast test, 12 assets | done, primary failed |
| 12 | Sampling-noise test (5 vs 20 sampled paths, all 438 BTC validation windows) | done, primary failed |
| 13a | Daily signal screen: momentum and reversal, 12 assets, no model | done; 1 marginal pass (see section 4) |

Key components:
- `src/kronosquant/`: `forecast.py`, `data.py`, `splits.py`, `baselines.py`, `metrics.py` (IC, bootstrap intervals, paired intervals), `evaluation.py`, `cache.py`, `backtest.py`, `strategy.py`, `finetune.py`, `crosssection.py`, `volatility.py`, `paper.py`, `dashboard_data.py`, `binance.py`.
- `scripts/`: one script per experiment; each states its **declared rule in the header** before running.
- `dashboard/app.py`, `dashboard/charts.py`, `.streamlit/config.toml`: read-only dashboard (scorecard, four-panel paper-account chart, result tables).
- `tests/`: 118 tests (last full run 117 passed; one later test for the paired IC interval passes on its own). Mutation checks were done on the leakage guard, the execution lag, and three paper-account safety rules.

Paper account (Stage 7): a strategy is a plain function that sees a **copy** of the past and returns a wish; a separate risk layer decides what is traded (max position, max order size, daily-loss halt that resets next day, permanent drawdown kill switch). Forced exits are never blocked and still pay costs. With no limit binding it matches the backtester to 1e-12. A halt acts at the next open, so losses can overshoot a limit.

Git state at writing: last pushed commit `d5a84f2` (Stage 8, first dashboard). **Uncommitted:** dashboard redesign (`dashboard/charts.py`, theme, `app.py`), the Stage 12 script, `paired_ic_diff_ci` in `metrics.py` with its test, `requirements.txt` (adds plotly 7.1.0), README and CLAUDE.md updates.

---

## 3. How the evidence was produced (why it can be trusted, and its limits)

- **Pre-declared rules** in each script header; reported pass/fail on the declared primary only; secondary results never used to rescue a failed primary.
- **99% bootstrap intervals** (many comparisons are made), resampling windows or dates.
- **Leakage controls:** forecast uses only rows before the origin (poison tests); baselines poison-tested; strategies see a copy of the past; the engine trades at the next open.
- **Reproducibility:** same runs reproduce bit-for-bit from the cache.
- **Test split untouched** throughout.

Known limits:
- One exchange (Binance), about 3 years of hourly data, one model size (Kronos-small), a hand-picked 12-asset universe (survivorship bias, unfixed).
- **Low statistical power.** The 99% IC interval is about +-0.16 on 300 windows, +-0.12 on 438 windows, and about +-0.09 for 12 assets x 100 dates. A real IC below about 0.1 could not have been detected.
- Costs in sweeps are assumed round numbers (5-20 bps). **Real Binance fees and terms of use are not verified.**
- The validation period was a falling market (BTC buy-and-hold about -41.6%): one regime.
- Kronos may have seen part of the train period in its own pretraining, so earlier history would flatter it; forward data is the only clean test.

---

## 4. Results so far (all on the validation split; none passed)

**Scorecard: 1 of 19 pre-declared primary comparisons passed** (5 grid variants, the Stage 6 strategy, 1 fine-tune, 4 cross-section signals, 3 volatility baselines, 3 daily signals, the 20-path test, the Stage 13b blend). The one pass is the marginal Stage 13a candidate XSMOM_28; one comparison (REV_5) is reliably worse; the other 17 are Inconclusive. Read with the dashboard's three states, **all 16 comparisons in the Kronos experiments are Inconclusive**: none passed and none was reliably worse, because the tests lacked the power to tell skill from luck either way (the secondary results, such as path error reliably above the random walk, are separate).

| Test | Headline (99% intervals) |
|---|---|
| Stage 3, HK stock (100 windows) | Did not beat the random walk on error; direction 49.0% vs 49.0% base rate; forecast correlated -0.51 with the previous move (reversal tilt) |
| Stage 3, BTC 1h (100 windows, 12h) | Did not beat the random walk; direction 54% vs 57% base rate; correlation with previous move about -0.53 |
| Grid, BTC 1h, 300 windows, IC | baseline -0.062 [-0.225, +0.096]; greedy -0.022 [-0.179, +0.135]; T=0.5 -0.068 [-0.230, +0.090]; lookback 128 +0.062 [-0.096, +0.202]; horizon 4 +0.098 [-0.076, +0.261]. Every variant had higher path error than the random walk |
| Stage 6, window strategy (299 windows) | Kronos long/short at 10 bps: -47.7% net (-23.2% at 0 bps); a random long/short null had a median -38.9% (73% of random strategies did at least as well); always long in windows -40.8%; buy-and-hold -41.6%. A contrarian rule made +19.7% gross but -24.2% net (exploratory, one bear market) |
| Fine-tune (BTC train split, 1000 steps) | Token loss improved (validation 2.3636 to 2.2229) but IC -0.054 [-0.211, +0.101] and path error reliably worse than the pretrained model |
| Stage 10, cross-section (12 assets x 100 dates) | Kronos mean rank IC -0.017 [-0.101, +0.076]; reversal baseline -0.003; momentum baseline +0.003; vol-scaled Kronos -0.047 |
| Stage 11, volatility (11 non-BTC assets) | Kronos error 0.307 vs 0.319 (history average), 0.288 (last 48 bars), 0.315 (same hour); every difference interval includes 0; Kronos best on 0 of 12 assets. Kronos tracks volatility (rank IC +0.49) but the last-48-bars baseline does at least as well (+0.54); Kronos forecasts ran about 10% too high |
| Stage 13a, daily signal screen (primary window 2019-2025-08, 10 bps, 99% block-bootstrap) | **XSMOM_28 passed marginally**: +46.7 bps per 5-day period [+1.0, +105.3]; TSMOM_28 inconclusive (+66.7 [-18.6, +162.1]); REV_5 reliably negative (-47.0 [-89.1, -11.6]). Edge is old: about +100 bps/period in 2019-2021, about +2 in 2022 to 2025-08, -1.1 in the recent window; interval includes 0 at 20 bps; long-only momentum earned less than equal-weight buy-and-hold (97.6 vs 129.2 bps); survivorship bias and free shorting are unfixed. Stage 13b (Kronos incremental on daily bars) was justified in the letter of the declared rule but the signal is weak |
| Stage 13b, Kronos on top of XSMOM_28 (242 periods, every second 5-day period 2019 to 2025-08, 10 bps, 99% block-bootstrap) | **Inconclusive**: blend minus baseline +18.8 bps per period [-32.7, +73.3] (half-width 53 bps). Blend +25.7, baseline +6.9, Kronos alone -13.7, all inconclusive. Rank IC difference +0.036 [-0.011, +0.084]. Kronos and momentum scores are negatively correlated (-0.11). The difference was positive in 2019 to 2022 and flat or negative in 2023 to 2025 (later years, more plausibly outside pretraining, though its cutoff is not known). A pass would only have been an upper bound (contamination); the baseline is weak on this half schedule; 2,813 forecasts took about 18 minutes, so the earlier 3 to 6 hour estimates were wrong |
| Stage 12, 5 vs 20 sampled paths (438 windows) | **Primary failed.** IC with 20 paths -0.0215 [-0.149, +0.118] vs -0.0387 [-0.169, +0.102] with 5; difference +0.017 [-0.029, +0.064]. The two forecasts correlate +0.953, so the 5-path forecast was already almost all signal and sampling noise does not explain the earlier results. Path error still reliably above the random walk; correlation with momentum -0.53; interval half-width 0.134 (an IC below that could not pass) |

Paper-account demo (mechanics, not a finding): BTC always-long with no limits -41.56% (matches the Stage 6 buy-and-hold); with limits (max position 0.5, max order 0.25, daily loss 3%, kill switch 15%) -11.9% net, -15.1% max drawdown, kill switch tripped in November 2025. It lost less only because it stopped trading in a falling market.

### Interpretation (best current reading)
1. **Strong evidence:** the target (12-hour crypto returns) is close to unpredictable; even simple baselines scored about zero.
2. **Strong evidence:** Kronos learned habits the data did not reward (a reversal tilt, correlation -0.37 to -0.52 with the previous move, in every test); averaging a few sampled paths adds error to a quantity whose best guess is near zero; better likelihood from fine-tuning did not mean better forecasts.
3. **Moderate evidence:** where Kronos does work (volatility tracking), a one-line baseline does equally well.
4. **Strong evidence:** the tests cannot see small effects.
5. **Ruled out:** leakage or engine bugs (tests, mutation checks, exact equivalence), costs as the only cause (strategies also lost gross).
6. **Unknown:** no signal at all vs a signal too faint to see; whether another market, bar size or Kronos-base behaves differently.

---

## 5. What accuracy would be needed to make money (back-of-envelope)

Computed from BTC validation data: 12-hour log-return standard deviation 1.62% (162 bps, n=438). For a sign strategy (Gaussian approximation, position flips half the time, a flip costs two one-way trades), the **break-even IC** is roughly:

| One-way cost | Break-even IC |
|---|---|
| 5 bps | about 0.04 |
| 10 bps | about 0.08 |
| 20 bps | about 0.16 |

Kronos measured between -0.06 and +0.10 with every interval including zero, and the current tests could not verify an IC of 0.08 anyway. Costs, not accuracy alone, are the main obstacle at this horizon. A preprint on hourly BTC reports the same pattern: sign strategies failed at 10 bps and a cost-aware execution filter restored profitability in selected configurations.

---

## 6. Market and landscape research (3 searches; vendor pages and papers; unverified)

- **Signal-audit tools already exist.** [AlphaAssay](https://mcpservers.org/servers/alphaassay/mcp) is a hosted validator (deflated Sharpe, probability of backtest overfitting, signed audit records, prepaid credits, no order path). [SYGNAL](https://api.sygnal.ai/signal-trading/about-signal-and-track-record-auditing.html) offers hash-based track-record auditing. A generic "backtest verifier" is partly taken. (Open-source tools such as Alphalens and mlfinlab are known from memory only; not verified by search.)
- **Independent evidence on finance foundation models agrees with our results.** [Rahimikia et al.](https://arxiv.org/html/2511.18578v1) found Chronos and TimesFM had negative out-of-sample R² on next-day returns (Chronos large -1.37%, TimesFM 500M -2.80%, linear baseline -0.47%) and direction accuracy just above 51%. They did not test Kronos.
- **Kronos's headline gains are the authors' own** (RankIC +93% vs the leading time-series foundation model, on their benchmarks) and I found no independent replication. These are *relative* gains over other models, not an absolute trading edge.
- **Forward-only, hash-anchored track records** are recognised as the credible evidence standard; backtests are hypotheses.

### Where something unique might be (hypotheses, not facts)
1. **An independent, forward-only, pre-registered scoreboard of finance foundation models** (Kronos, Chronos, TimesFM, Moirai vs simple baselines). I found no one publishing this for Kronos.
2. **Open and local:** hosted tools charge per call and run remotely; this would run on the user's machine with the user's rules.
3. **Minimum detectable effect and minimum-track-record calculators** plus a pre-registration log (already built in spirit). Not confirmed absent from competitors.
4. **Possible market focus**, e.g. India with local costs. Data licensing and regulation are unchecked.

---

## 7. Viability (judgement, low evidence)

| Goal | Viability | Why |
|---|---|---|
| Learning and portfolio | High | Already built, tested, documented |
| Respected open-source verification tool or Kronos benchmark | Moderate | Technically close; depends on whether anyone cares |
| Paid product | Low to uncertain | Competitors exist; signal buyers often want promises, not verification; quants often build their own tools. Do not build for revenue before demand is shown |
| Trading profits | Not supported on current evidence | Eight experiments found no usable Kronos edge |

Strengths: low technical risk; small running cost at small scale (Kronos-small forward forecasts for a few dozen assets need minutes of compute per hour on this Mac, about 4 s per forecast; bigger models need a cloud machine, not priced). Weaknesses: unknown demand, crowded neighbours, negative results are hard to market, a forward ledger needs months of steady upkeep.

**Cheapest viability test:** publish a write-up of the independent Kronos check, talk to 5-10 likely users, and decide *in advance* what counts as a yes (suggested: at least 3 of 10 say they would use it, and a handful of strangers run it; these thresholds are a suggestion, not a standard).

---

## 8. How measurement could become profit (research summary)

Measurement does not create profit; it screens out ideas that lose money. Profit needs an edge that survives costs. Evidence on where returns have been reported (read from search summaries, not full papers; several are preprints or working papers):

| Route | Evidence | Caveats |
|---|---|---|
| Crypto time-series momentum (daily/weekly) | [Han, Kang, Ryu](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf): Sharpe 1.51 (28-day lookback, 5-day hold) vs 0.84 for the market, costs from real fees | Profits concentrated in rising markets; many become insignificant after costs; liquidations common |
| ML across many coins with costs | [Fischer et al.](https://open.fau.de/items/fa78c545-184e-4239-aeb4-b43ef33f467d/full): 7.1 bps/day after 15 bps costs (2018). [Jaquart et al.](https://bura.brunel.ac.uk/bitstream/2438/29964/4/FullText.pdf): out-of-sample Sharpe about 3.1-3.2 after costs | Short samples; limits to arbitrage not ruled out |
| Funding-rate and basis carry | [Survey](https://arxiv.org/pdf/2510.14435): Sharpe 6.45 (2020-25), about 4.06 from 2024, negative in 2025. [He et al.](https://arxiv.org/pdf/2212.06888): high Sharpe even at top Binance costs | Decaying; capacity limited by venue liquidity; needs derivatives access that may not be legal or available from India (unchecked) |
| Volatility-managed sizing | Strong in-sample | [Cederburg et al.](https://alphaarchitect.com/does-portfolio-timing-based-on-volatility-signals-outperform-buy-and-hold/): real-time versions often do not beat simple holding |

Why most claims vanish: [McLean and Pontiff](https://ivey.uwo.ca/media/3775549/pontiff.pdf) (97 published predictors): returns are 26% lower out of sample and 58% lower after publication. A currency study found technical-rule Sharpe falling from 0.66 in-sample to 0.06 out-of-sample.

Breadth: the textbook "fundamental law" (information ratio about IC x square root of breadth; from memory, not searched) says a tiny IC repeated across many independent bets can matter. Crypto assets are highly correlated, so effective breadth is far below the asset count.

Retail context ([SEBI FY25 study](https://www.business-standard.com/markets/news/net-losses-of-traders-in-fo-widens-in-fy25-sebi-study-125070701221_1.html)): over 91% of individual F&O traders lost money; net losses Rs 105,603 crore (+41% on FY24); only about 1 in 200 stayed profitable every year. A tool that stops people trading unverified ideas has user value; presenting it to others as advice raises regulatory questions that have not been checked.

---

## 9. Plan

### 9.1 Proposed next experiments ("signal factory"; awaiting the owner's go-ahead)
1. **Stage 12 (sampling noise) is finished**: it failed; sampling noise is not the explanation (see section 4).
2. **Stage 13, signal screen** (13a and 13b done, see section 4; 13b was run at every second period and is inconclusive; a full run is about 40 minutes of CPU): pre-declared test of time-series momentum, cross-sectional momentum, reversal, carry (needs funding-rate data; availability not verified) and Kronos as one input, on **daily bars** across the 12 assets, using **longer history back to 2017** (classical signals have no pretraining-contamination problem, so more history is usable, which adds power). Count of signals tried is declared and corrected for. Cost-aware thresholds and low turnover from the start. Validation only; test split opened once, after a pass.
3. **Stacking test:** fit a small regularised model on train-period forecasts (Kronos, momentum, reversal, recent volatility) and evaluate it unchanged on validation: does Kronos add anything beyond simple rules? Cost: about 20 min per asset of train-split forecasting.
4. **Store individual sample paths** (the cache keeps only the mean) and score probabilities and quantiles, not just the mean.
5. **Other untested levers:** Kronos-base (102M; slower, needs a pinned revision), fine-tuning on all 12 assets (about 12x the data), shorter horizons (1-4 bars; best IC so far appeared at 4), other markets (daily stocks; needs a data-source check).
6. **Risk-side value:** volatility-targeted sizing and lower turnover can improve drawdown and Sharpe for any strategy; evidence for real-time gains is mixed (Cederburg et al.).
7. Anything that passes validation goes to **at least 3 months of forward paper trading** before any claim. Real-money use is outside this project's rules and would be the owner's separate decision.

### 9.2 Scaling path toward a useful product
| Phase | Build | Gate |
|---|---|---|
| 0. Decide (1-2 weeks) | Verify competitors' features; talk to 5-10 likely users; one-page positioning; publish the independent Kronos write-up | At least 3 of 10 would use or contribute; otherwise stay a research and learning project |
| 1. Generalise (weeks 2-6) | Model-agnostic audit command (forecasts + prices in, report out: IC with interval, baselines, power, costs, multiple-testing adjustment); adapters for Chronos, TimesFM, Moirai; pip package; protocol file | Reproduces current results; an outsider audits their own CSV in about 10 minutes |
| 2. Forward ledger (weeks 4-10) | Scheduled forecasts for N assets and models; hash-chained, publicly anchored log written before outcomes; public scoreboard | About 3 months of forward data before any claim |
| 3. Breadth | Daily stocks, FX, maybe NSE; cloud or GPU machine | Costs and data licences verified per market |
| 4. Surfaces | Hosted report or API; alerts when live IC leaves its interval; paper account as a "risk copilot" | Only if phases 0-2 showed demand |

---

## 10. Risks and unknowns

- **Regulation:** research tools can drift into advice (SEBI research-analyst rules in India, US equivalents). Not checked.
- **Data licences and terms:** Binance terms unverified; exchange data for NSE is usually licensed; funding-rate data availability and legality of crypto derivatives from India unchecked.
- **Pretraining contamination:** forward data is the only clean test for Kronos.
- **Most candidate signals will fail**, and any pass is likely small, capacity-limited and decaying.
- **Attention:** a forward ledger needs months of upkeep.
- **Research quality:** market research above rests on search summaries, not full-paper reads; several sources are preprints or working papers; the Harvey-Liu-Zhu multiple-testing paper was not retrieved.

---

## 11. Open decisions for the owner

1. **First user:** the owner, retail traders buying AI signals, or researchers benchmarking models?
2. **Aim:** open-source credibility or a paid product?
3. **Market focus:** crypto only, or India equities (data and licensing to be checked)?
4. **Go-ahead** for Stage 13 (signal screen), the stacking test, Kronos-base, or the write-up first?
5. Whether this file and the product strategy should live in the public repo.

---

## 12. How to run things (from the repo root)

```bash
.venv/bin/python -I -W ignore::DeprecationWarning -m pytest -q          # tests
.venv/bin/python -I -m streamlit run dashboard/app.py                   # dashboard (restart after editing charts.py)
.venv/bin/python -I -W ignore scripts/stage12_samples.py                # Stage 12 (long; resumable via the cache)
```

Always run interpreters with `-I` from the repo root, never from inside `vendor/Kronos/`. The built-in preview tool reads a launch config from the parent folder (another project's), so start Streamlit with the shell and open `http://localhost:8501` by URL.

---

## 13. Sources

- Kronos: [repository](https://github.com/shiyu-coder/Kronos), [paper](https://arxiv.org/html/2508.02739v1)
- [AlphaAssay](https://mcpservers.org/servers/alphaassay/mcp), [SYGNAL track-record auditing](https://api.sygnal.ai/signal-trading/about-signal-and-track-record-auditing.html)
- [Rahimikia et al., Re(Visiting) Time Series Foundation Models in Finance](https://arxiv.org/html/2511.18578v1)
- [Han, Kang, Ryu, crypto momentum](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/918729/Time_Series_and_Cross_Sectional_Momentum_in_the_Cryptocurrency_Market_with_IA.pdf)
- [Fischer et al., statistical arbitrage in crypto](https://open.fau.de/items/fa78c545-184e-4239-aeb4-b43ef33f467d/full), [Jaquart et al.](https://bura.brunel.ac.uk/bitstream/2438/29964/4/FullText.pdf), [hourly BTC walk-forward preprint](https://www.emergentmind.com/papers/2606.00060)
- [Crypto carry survey](https://arxiv.org/pdf/2510.14435), [Fundamentals of Perpetual Futures](https://arxiv.org/pdf/2212.06888)
- [Cederburg et al. via AlphaArchitect](https://alphaarchitect.com/does-portfolio-timing-based-on-volatility-signals-outperform-buy-and-hold/), [McLean and Pontiff](https://ivey.uwo.ca/media/3775549/pontiff.pdf)
- [SEBI FY25 F&O study via Business Standard](https://www.business-standard.com/markets/news/net-losses-of-traders-in-fo-widens-in-fy25-sebi-study-125070701221_1.html)
