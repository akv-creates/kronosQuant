"""Stage 2: load and validate OHLCV candles. Bad data should fail loudly, not silently skew results."""
from pathlib import Path

import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]
REQUIRED_COLS = ["timestamps", *PRICE_COLS, "volume"]


class DataError(ValueError):
    """Raised when candle data breaks a basic assumption of the pipeline."""


def validate_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Check the invariants every later stage relies on. Returns df unchanged if all hold."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DataError(f"missing columns: {missing}")
    if df[REQUIRED_COLS].isna().any().any():
        raise DataError("NaN values present")
    ts = df["timestamps"]
    if not ts.is_monotonic_increasing:
        raise DataError("timestamps are not sorted")
    if ts.duplicated().any():
        raise DataError("duplicate timestamps")
    if (df[PRICE_COLS] <= 0).any().any():
        raise DataError("non-positive prices")
    if (df["volume"] < 0).any():
        raise DataError("negative volume")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise DataError("high is below open/close/low on some rows")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise DataError("low is above open/close/high on some rows")
    return df


def find_gaps(df: pd.DataFrame, bar: str = "5min") -> pd.DataFrame:
    """Rows whose gap to the previous candle is longer than one bar (lunch, overnight, weekends, outages).

    Gaps are normal for equities, so they are reported, not rejected. For 24/7 markets like crypto,
    any gap is suspicious.
    """
    step = df["timestamps"].diff()
    mask = step > pd.Timedelta(bar)
    return pd.DataFrame({"timestamps": df.loc[mask, "timestamps"], "gap": step[mask]})


def load_candles(path: Path) -> pd.DataFrame:
    """Read a candle CSV (BOM-safe), parse timestamps, sort chronologically, validate."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df = df.sort_values("timestamps").reset_index(drop=True)
    return validate_candles(df)


def has_gap(df: pd.DataFrame, rows: range, bar: str = "5min") -> bool:
    """True if any consecutive pair of candles inside `rows` is more than one bar apart."""
    ts = df["timestamps"].iloc[rows.start : rows.stop]
    return bool((ts.diff().dropna() > pd.Timedelta(bar)).any())
