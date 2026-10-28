"""Fetch historical candles from Binance's public market-data endpoint. No API key, no account, read-only.

Timestamp convention: Binance labels a candle by its OPEN time (a 10:00 hourly bar covers 10:00-11:00 and is only
known at 11:00). Stored timestamps are UTC, timezone-naive. Only fully closed candles are kept: an in-progress
last candle would silently leak its not-yet-final values into the data.
"""
import time

import pandas as pd
import requests

BASE_URL = "https://data-api.binance.vision"  # Binance's market-data-only host
KLINES_PATH = "/api/v3/klines"
INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
INTERVAL_PANDAS = {"5m": "5min", "15m": "15min", "1h": "1h"}
PAGE_LIMIT = 1000  # the API's maximum candles per request


def http_get_json(url: str, params: dict, *, retries: int = 5, timeout: float = 20, sleep=time.sleep):
    """GET with backoff on rate limits (429/418) and server errors. Honours Retry-After."""
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (418, 429) or r.status_code >= 500:
            sleep(float(r.headers.get("Retry-After", 2**attempt)))
            continue
        r.raise_for_status()
    raise RuntimeError(f"giving up after {retries} attempts (last status {r.status_code})")


def fetch_klines(get_json, symbol: str, interval: str, start_ms: int, end_ms: int, pause: float = 0.2, sleep=time.sleep) -> list:
    """All raw kline rows with open time in [start_ms, end_ms), fetched page by page."""
    step = INTERVAL_MS[interval]
    rows, cursor = [], start_ms
    while cursor < end_ms:
        page = get_json(
            BASE_URL + KLINES_PATH,
            {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms - 1, "limit": PAGE_LIMIT},
        )
        if not page:
            break
        rows.extend(page)
        nxt = page[-1][0] + step
        if nxt <= cursor:
            raise RuntimeError("pagination is not advancing")
        cursor = nxt
        if len(page) < PAGE_LIMIT:
            break
        sleep(pause)
    return rows


def klines_to_candles(rows: list, now_ms: int) -> pd.DataFrame:
    """Raw rows -> our schema. Drops candles that have not closed yet and duplicate open times."""
    df = pd.DataFrame(
        [
            {
                "open_ms": int(r[0]),
                "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]),
                "volume": float(r[5]),
                "amount": float(r[7]),  # quote-asset volume (turnover in USDT), Kronos's `amount` column
                "close_ms": int(r[6]),
            }
            for r in rows
        ]
    )
    if df.empty:
        raise ValueError("no candles returned")
    df = df[df["close_ms"] < now_ms].drop_duplicates("open_ms").sort_values("open_ms")
    df["timestamps"] = pd.to_datetime(df["open_ms"], unit="ms")
    return df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].reset_index(drop=True)
