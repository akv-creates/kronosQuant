import pandas as pd
import pytest

from kronosquant.binance import INTERVAL_MS, PAGE_LIMIT, fetch_klines, http_get_json, klines_to_candles

H = INTERVAL_MS["1h"]


def row(open_ms, o=1.0, h=2.0, l=0.5, c=1.5, v=10.0, q=15.0):
    # Binance kline layout: open time, o, h, l, c, volume, close time, quote volume, trades, ...
    return [open_ms, str(o), str(h), str(l), str(c), str(v), open_ms + H - 1, str(q), 5, "0", "0", "0"]


class FakeApi:
    """Serves hourly candles from a fixed list, honouring startTime/endTime/limit like the real endpoint."""

    def __init__(self, n):
        self.rows = [row(i * H) for i in range(n)]
        self.calls = []

    def __call__(self, url, params):
        self.calls.append(params["startTime"])
        out = [r for r in self.rows if params["startTime"] <= r[0] <= params["endTime"]]
        return out[: params["limit"]]


def test_pagination_collects_every_candle_exactly_once_across_pages():
    api = FakeApi(2500)
    rows = fetch_klines(api, "BTCUSDT", "1h", 0, 2500 * H, sleep=lambda s: None)
    assert [r[0] for r in rows] == [i * H for i in range(2500)]
    assert len(api.calls) == 3 and api.calls[1] == PAGE_LIMIT * H  # cursor advances past the last row


def test_fetch_respects_the_requested_range():
    rows = fetch_klines(FakeApi(100), "BTCUSDT", "1h", 10 * H, 20 * H, sleep=lambda s: None)
    assert [r[0] for r in rows] == [i * H for i in range(10, 20)]


def test_non_advancing_pagination_raises_instead_of_looping_forever():
    stuck = lambda url, params: [row(0)] * PAGE_LIMIT  # noqa: E731
    with pytest.raises(RuntimeError, match="not advancing"):
        fetch_klines(stuck, "BTCUSDT", "1h", 5 * H, 10**12, sleep=lambda s: None)


def test_incomplete_last_candle_is_dropped_and_schema_is_ours():
    rows = [row(0), row(H), row(2 * H)]
    now_ms = 2 * H + 1000  # candle 2 is still open; candles 0 and 1 have closed
    df = klines_to_candles(rows, now_ms)
    assert len(df) == 2
    assert list(df.columns) == ["timestamps", "open", "high", "low", "close", "volume", "amount"]
    assert df["timestamps"].iloc[1] == pd.Timestamp("1970-01-01 01:00:00")  # open time, UTC, tz-naive
    assert df["amount"].iloc[0] == 15.0  # quote volume mapped to `amount`


def test_duplicates_are_removed_and_output_is_sorted():
    df = klines_to_candles([row(H), row(0), row(H)], now_ms=10 * H)
    assert df["timestamps"].is_monotonic_increasing and len(df) == 2


def test_empty_response_is_an_error():
    with pytest.raises(ValueError, match="no candles"):
        klines_to_candles([], now_ms=1)


class Resp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._p, self.headers = status, payload, headers or {}

    def json(self):
        return self._p

    def raise_for_status(self):
        raise RuntimeError(f"HTTP {self.status_code}")


def test_http_retries_on_rate_limit_and_honours_retry_after(monkeypatch):
    responses = iter([Resp(429, headers={"Retry-After": "7"}), Resp(503), Resp(200, [1, 2])])
    monkeypatch.setattr("kronosquant.binance.requests.get", lambda *a, **k: next(responses))
    waits = []
    assert http_get_json("u", {}, sleep=waits.append) == [1, 2]
    assert waits == [7.0, 2.0]  # Retry-After first, then exponential backoff (2**1)


def test_http_gives_up_after_retries_and_does_not_retry_client_errors(monkeypatch):
    monkeypatch.setattr("kronosquant.binance.requests.get", lambda *a, **k: Resp(429))
    with pytest.raises(RuntimeError, match="giving up"):
        http_get_json("u", {}, retries=3, sleep=lambda s: None)
    calls = []
    monkeypatch.setattr("kronosquant.binance.requests.get", lambda *a, **k: calls.append(1) or Resp(400))
    with pytest.raises(RuntimeError, match="HTTP 400"):
        http_get_json("u", {}, sleep=lambda s: None)
    assert len(calls) == 1
