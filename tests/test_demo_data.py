import io

import numpy as np
import pandas as pd
import pytest

from kronosquant.data import DataError, validate_candles
from kronosquant.demo_data import MARKETS, MAX_UPLOAD_ROWS, parse_uploaded_candles, synthetic_candles
from kronosquant.paper import RiskLimits, run_paper


@pytest.mark.parametrize("kind", list(MARKETS))
def test_every_synthetic_market_is_valid_hourly_data(kind):
    c = synthetic_candles(kind, n_bars=500, seed=3)
    validate_candles(c)  # raises on any broken candle
    assert len(c) == 500 and (c["timestamps"].diff().dropna() == pd.Timedelta("1h")).all()
    assert np.allclose(c["open"].iloc[1:].to_numpy(), c["close"].iloc[:-1].to_numpy())  # each bar opens at the previous close


def test_synthetic_markets_are_deterministic_and_differ_by_seed_and_kind():
    a, b = synthetic_candles(seed=1), synthetic_candles(seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["close"], synthetic_candles(seed=2)["close"])
    assert not np.allclose(a["close"], synthetic_candles("Steady uptrend", seed=1)["close"])


def test_trend_and_crash_markets_behave_as_described():
    up, down = synthetic_candles("Steady uptrend", n_bars=4000, seed=5), synthetic_candles("Steady downtrend", n_bars=4000, seed=5)
    assert up["close"].iloc[-1] > 100 > down["close"].iloc[-1]  # drift of +-0.0004 per bar over 4000 bars dwarfs the noise
    crash = synthetic_candles("Crash then recovery", n_bars=2000, seed=5)["close"].to_numpy()
    assert crash[int(2000 * 0.4) + 150] < 0.8 * crash[int(2000 * 0.4) - 1]  # the sharp fall really happened
    chop = synthetic_candles("Choppy and mean-reverting", n_bars=4000, seed=5)["close"]
    assert np.log(chop).std() < np.log(synthetic_candles(n_bars=4000, seed=5)["close"]).std()  # snaps back, so it wanders less than a random walk


def test_synthetic_input_is_validated():
    with pytest.raises(ValueError):
        synthetic_candles(n_bars=10)
    with pytest.raises(ValueError):
        synthetic_candles(volatility=0)
    with pytest.raises(ValueError):
        synthetic_candles("not a market")


def test_the_paper_account_runs_on_synthetic_data():
    res = run_paper(synthetic_candles(n_bars=800), lambda history, position: 1.0, limits=RiskLimits(max_position=0.5, max_order=0.25, max_daily_loss=0.5, max_drawdown=0.9), fee_bps=10, slippage_bps=0)
    assert len(res.frame) == 800 and np.isfinite(res.metrics["net_return"])


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def test_upload_accepts_a_good_file_with_aliases_and_missing_volume_and_sorts_it():
    c = synthetic_candles(n_bars=200).rename(columns={"timestamps": "Time"}).drop(columns=["volume"]).iloc[::-1]  # reversed, odd header, no volume
    out = parse_uploaded_candles(csv_bytes(c))
    assert list(out.columns) == ["timestamps", "open", "high", "low", "close", "volume"] and out["timestamps"].is_monotonic_increasing and (out["volume"] == 0).all()
    assert len(out) == 200


def test_upload_converts_timezone_aware_times_to_naive_utc():
    c = synthetic_candles(n_bars=150)
    c["timestamps"] = c["timestamps"].dt.tz_localize("UTC").dt.tz_convert("Asia/Kolkata").astype(str)
    out = parse_uploaded_candles(io.BytesIO(csv_bytes(c)))
    assert out["timestamps"].iloc[0] == pd.Timestamp("2024-01-01 00:00") and out["timestamps"].dt.tz is None


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.drop(columns=["close"]), "missing column"),
    (lambda d: d.iloc[:50], "at least"),
    (lambda d: d.assign(close=np.nan), "NaN"),
    (lambda d: d.assign(high=d["low"] - 1), "high"),
    (lambda d: d.assign(open="abc"), "number"),
    (lambda d: d.assign(timestamps="not a date"), "date"),
    (lambda d: d.assign(open=-1.0), "non-positive"),
    (lambda d: pd.concat([d, d.iloc[:1]]), "duplicate"),
])
def test_upload_rejects_bad_files_with_a_clear_message(mutate, message):
    good = synthetic_candles(n_bars=200)
    with pytest.raises(DataError, match=message):
        parse_uploaded_candles(csv_bytes(mutate(good)))


def test_upload_rejects_infinity_oversized_and_unreadable_files():
    c = synthetic_candles(n_bars=200)
    c.loc[5, "close"] = np.inf
    with pytest.raises(DataError, match="finite|high"):
        parse_uploaded_candles(csv_bytes(c))
    with pytest.raises(DataError, match="limit"):
        parse_uploaded_candles(csv_bytes(synthetic_candles(n_bars=300)), max_rows=200)
    with pytest.raises(DataError, match="MB"):
        parse_uploaded_candles(b"x" * 100, max_bytes=10)
    with pytest.raises(DataError, match="CSV"):
        parse_uploaded_candles(b"")
    assert MAX_UPLOAD_ROWS >= 5260  # the real validation split must fit in the same limit
