import pandas as pd

from kronosquant import forecast
from kronosquant.config import CONFIDENCE_LEVEL, KRONOS_V0, UNIVERSE
from kronosquant.splits import Window


def test_shared_research_settings_are_frozen():
    """Experiments declare their rule before running; changing one of these values must be a deliberate, visible edit."""
    assert KRONOS_V0 == dict(lookback=400, pred_len=12, T=1.0, top_p=0.9, top_k=0, sample_count=5)
    assert CONFIDENCE_LEVEL == 0.99
    assert UNIVERSE == ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LTCUSDT", "TRXUSDT", "LINKUSDT", "BCHUSDT", "ETCUSDT", "XLMUSDT"]


def test_make_forecaster_loads_lazily_reuses_the_model_and_seeds_with_the_origin(monkeypatch):
    loads, calls = [], []
    monkeypatch.setattr(forecast, "_PREDICTORS", {})
    monkeypatch.setattr(forecast, "load_predictor", lambda device, ckpt=None: loads.append((device, ckpt)) or "model")
    monkeypatch.setattr(forecast, "forecast_window", lambda predictor, candles, origin, lookback, pred_len, **kw: calls.append((predictor, origin, lookback, pred_len, kw)) or pd.DataFrame())
    fn = forecast.make_forecaster(pd.DataFrame(), "cpu", T=0.5, top_p=0.8, sample_count=3, top_k=2)
    assert loads == []  # building the forecaster must not load the model
    fn(Window(origin=500, lookback=400, pred_len=12))
    fn(Window(origin=512, lookback=400, pred_len=12))
    assert loads == [("cpu", None)]  # loaded once, reused
    assert calls[0][:4] == ("model", 500, 400, 12) and calls[1][1] == 512
    assert calls[0][4] == dict(seed=500, T=0.5, top_p=0.8, top_k=2, sample_count=3)  # seed policy: seed = window origin
