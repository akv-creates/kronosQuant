"""Settings shared by the experiment scripts, so each value is defined once.

Changing a value here changes every experiment that uses it, and `tests/test_config.py` pins the current values on purpose:
experiments declare their rule before they run, so a silent change must fail a test instead of altering a result.
Nothing here imports torch, so the dashboard and the analysis scripts can use it without loading the model.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Twelve large Binance USDT pairs, picked in advance for full history and no gaps. The list was chosen today, so assets that were
# delisted or collapsed are absent (survivorship bias); that flaw is known and not corrected.
UNIVERSE = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "TRXUSDT", "LINKUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT"]

# The default Kronos inference settings ("V0") used by the hourly experiments: 400 bars of history, forecast 12 bars,
# sampling temperature 1.0, nucleus sampling 0.9, no top-k cut, average of 5 sampled paths.
KRONOS_V0 = dict(lookback=400, pred_len=12, T=1.0, top_p=0.9, top_k=0, sample_count=5)

# Confidence level of every primary interval (99%), chosen because several candidates are tried in each experiment.
CONFIDENCE_LEVEL = 0.99
