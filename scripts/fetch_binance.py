"""Download closed Binance candles into data/ (git-ignored), validate them, and report gaps. No keys involved."""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kronosquant.binance import INTERVAL_MS, INTERVAL_PANDAS, fetch_klines, http_get_json, klines_to_candles  # noqa: E402
from kronosquant.cache import data_fingerprint  # noqa: E402
from kronosquant.data import find_gaps, validate_candles  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", default="BTCUSDT")
parser.add_argument("--interval", default="1h", choices=sorted(INTERVAL_MS))
parser.add_argument("--years", type=float, default=3.0)
parser.add_argument("--end", help="UTC date YYYY-MM-DD (exclusive). Default: today 00:00 UTC, so same-day reruns give identical data.")
args = parser.parse_args()

today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else today
start = end - timedelta(days=round(365.25 * args.years))
start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
print(f"fetching {args.symbol} {args.interval}: {start:%Y-%m-%d} .. {end:%Y-%m-%d} (UTC)")

t0 = time.time()
rows = fetch_klines(http_get_json, args.symbol, args.interval, start_ms, end_ms)
df = validate_candles(klines_to_candles(rows, now_ms=int(time.time() * 1000)))
print(f"{len(df)} candles in {time.time() - t0:.0f}s: {df['timestamps'].iloc[0]} .. {df['timestamps'].iloc[-1]}")

gaps = find_gaps(df, INTERVAL_PANDAS[args.interval])
print(f"gaps longer than one bar: {len(gaps)}")
if len(gaps):
    print(gaps.head(10).to_string(index=False))

out = ROOT / "data"
out.mkdir(exist_ok=True)
csv = out / f"{args.symbol}_{args.interval}.csv"
df.to_csv(csv, index=False)
meta = {
    "source": "Binance public klines (data-api.binance.vision)",
    "symbol": args.symbol, "interval": args.interval,
    "requested_utc": [start.isoformat(), end.isoformat()],
    "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "rows": len(df), "gaps": len(gaps), "fingerprint": data_fingerprint(df),
    "timestamp_convention": "candle OPEN time, UTC, tz-naive",
}
csv.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
print("saved", csv)
