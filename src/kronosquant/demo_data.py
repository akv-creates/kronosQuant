"""Demo candles for the interactive paper-account simulator on the hosted dashboard: simulated markets and visitor-uploaded files.

Why these exist: the research results use real Binance candles, which cannot be redistributed, so the hosted simulator runs on data that is either
made up here (seeded, so a seed always gives the same market) or supplied by the visitor and processed in memory only. Demo data is never
mixed into the research results, and none of it touches the sealed test split.
"""
import io

import numpy as np
import pandas as pd

from kronosquant.data import DataError, validate_candles

MAX_UPLOAD_ROWS = 20_000  # about 2.3 years of hourly bars; keeps a run (and the cost sweep on the page) quick on a small server
MAX_UPLOAD_BYTES = 5_000_000
MIN_ROWS = 100
COLUMN_ALIASES = {"timestamp": "timestamps", "time": "timestamps", "date": "timestamps", "datetime": "timestamps", "open_time": "timestamps"}
PRICE_COLUMNS = ["open", "high", "low", "close"]

MARKETS = {
    "Random walk (no drift)": "Prices move up and down by chance with no overall direction.",
    "Steady uptrend": "A random walk with a small upward drift.",
    "Steady downtrend": "A random walk with a small downward drift.",
    "Choppy and mean-reverting": "Prices keep snapping back toward a starting level, so trends do not last.",
    "Crash then recovery": "A random walk with a sharp fall about 40% of the way through, followed by a slow recovery.",
}


def _log_returns(kind: str, n: int, volatility: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, volatility, n)
    if kind == "Random walk (no drift)":
        return noise
    if kind == "Steady uptrend":
        return noise + 0.0004
    if kind == "Steady downtrend":
        return noise - 0.0004
    if kind == "Choppy and mean-reverting":
        returns, deviation = np.empty(n), 0.0  # deviation of log price from its starting level pulls the next return back
        for t in range(n):
            returns[t] = -0.03 * deviation + noise[t]
            deviation += returns[t]
        return returns
    if kind == "Crash then recovery":
        drift = np.full(n, 0.0005)
        start = int(n * 0.4)
        drift[start : start + 150] = -0.0030
        return noise + drift
    raise ValueError(f"unknown market {kind!r}; choose one of {list(MARKETS)}")


def synthetic_candles(kind: str = "Random walk (no drift)", n_bars: int = 4000, seed: int = 0, volatility: float = 0.005) -> pd.DataFrame:
    """Hourly candles for a simulated market. Deterministic for a given (kind, n_bars, seed, volatility); passes the same validation as real data.

    `volatility` is the standard deviation of one bar's log return (0.005 = 0.5% per hour, roughly crypto-sized)."""
    if n_bars < MIN_ROWS or n_bars > MAX_UPLOAD_ROWS:
        raise ValueError(f"n_bars must be between {MIN_ROWS} and {MAX_UPLOAD_ROWS}")
    if not 0 < volatility < 0.2:
        raise ValueError("volatility must be between 0 and 0.2")
    rng = np.random.default_rng(seed)
    returns = _log_returns(kind, n_bars, volatility, rng)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[100.0], close[:-1]])  # each bar opens where the last one closed
    wick_up, wick_down = np.abs(rng.normal(0.0, volatility / 2, n_bars)), np.abs(rng.normal(0.0, volatility / 2, n_bars))
    candles = pd.DataFrame({
        "timestamps": pd.date_range("2024-01-01", periods=n_bars, freq="h"),
        "open": open_, "high": np.maximum(open_, close) * np.exp(wick_up), "low": np.minimum(open_, close) * np.exp(-wick_down), "close": close,
        "volume": rng.lognormal(mean=5.0, sigma=0.5, size=n_bars),
    })
    return validate_candles(candles)


def parse_uploaded_candles(file, max_rows: int = MAX_UPLOAD_ROWS, max_bytes: int = MAX_UPLOAD_BYTES) -> pd.DataFrame:
    """Read an uploaded CSV (bytes or a file-like object) into validated candles, or raise `DataError` with a message a visitor can act on.

    Needs a time column (timestamps, timestamp, time, date or datetime) and open, high, low, close; volume is optional (treated as 0).
    The file is read in memory and never stored."""
    raw = file if isinstance(file, (bytes, bytearray)) else file.read()
    if len(raw) > max_bytes:
        raise DataError(f"the file is {len(raw) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.0f} MB")
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as err:
        raise DataError(f"could not read the file as a CSV ({type(err).__name__})") from err
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_ALIASES)
    missing = [c for c in ["timestamps", *PRICE_COLUMNS] if c not in df.columns]
    if missing:
        raise DataError(f"missing column(s): {', '.join(missing)}. Found: {', '.join(df.columns[:8])}")
    if len(df) < MIN_ROWS:
        raise DataError(f"only {len(df)} rows; at least {MIN_ROWS} are needed")
    if len(df) > max_rows:
        raise DataError(f"{len(df):,} rows; the limit is {max_rows:,}. Upload a shorter, more recent slice")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    try:
        stamps = pd.to_datetime(df["timestamps"], utc=True)
        df["timestamps"] = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
        for col in [*PRICE_COLUMNS, "volume"]:
            df[col] = pd.to_numeric(df[col])
    except (ValueError, TypeError) as err:
        raise DataError(f"a time or price value could not be read as a date or number ({type(err).__name__})") from err
    df = df[["timestamps", *PRICE_COLUMNS, "volume"]].sort_values("timestamps").reset_index(drop=True)
    if not np.isfinite(df[[*PRICE_COLUMNS, "volume"]].to_numpy(float)).all():
        raise DataError("some prices or volumes are not finite numbers (NaN or infinity)")
    return validate_candles(df)
