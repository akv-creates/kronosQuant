"""Stage 1: load pretrained Kronos-small and produce one forecast."""
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "vendor" / "Kronos"))  # upstream repo is cloned, not pip-installed

from kronosquant.data import load_candles  # noqa: E402
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402

# Hugging Face revisions pinned to the ones upstream's own regression test uses,
# so a later upstream weight update cannot silently change our results.
TOKENIZER_ID = "NeoQuasar/Kronos-Tokenizer-base"
TOKENIZER_REVISION = "0e0117387f39004a9016484a186a908917e22426"
MODEL_ID = "NeoQuasar/Kronos-small"
MODEL_REVISION = "901c26c1332695a2a8f243eb2f37243a37bea320"
MAX_CONTEXT = 512

# 5-minute Alibaba (HK: 09988) candles shipped inside the upstream repo.
SAMPLE_CSV = ROOT / "vendor" / "Kronos" / "finetune_csv" / "data" / "HK_ali_09988_kline_5min_all.csv"
FEATURES = ["open", "high", "low", "close", "volume", "amount"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_sample_candles(path: Path = SAMPLE_CSV) -> pd.DataFrame:
    return load_candles(path)


def load_predictor(device: str = "cpu") -> KronosPredictor:
    tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    model = Kronos.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.eval()
    model.eval()
    return KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)


def forecast_window(
    predictor: KronosPredictor,
    candles: pd.DataFrame,
    origin: int,
    lookback: int = 400,
    pred_len: int = 120,
    seed: int = 0,
    T: float = 1.0,
    top_p: float = 0.9,
    sample_count: int = 1,
) -> pd.DataFrame:
    """Forecast `pred_len` candles that start at row `origin`, using only rows before it.

    The model sees rows [origin - lookback, origin). Rows from `origin` on are used only
    for their timestamps (the model needs to know *when* each future candle happens).
    """
    if origin < lookback or origin + pred_len > len(candles):
        raise ValueError("window does not fit inside the data")
    history = candles.iloc[origin - lookback : origin]
    future = candles.iloc[origin : origin + pred_len]
    set_seed(seed)
    with torch.no_grad():
        return predictor.predict(
            df=history[FEATURES].reset_index(drop=True),
            x_timestamp=history["timestamps"].reset_index(drop=True),
            y_timestamp=future["timestamps"].reset_index(drop=True),
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )
