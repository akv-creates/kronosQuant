"""Stage 4: a resumable on-disk cache for expensive forecasts. Model-free, so backtests can read it without PyTorch."""
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kronosquant.splits import Window


def data_fingerprint(candles: pd.DataFrame) -> str:
    """Short hash of the candle values. Any changed, added or removed row changes it."""
    h = hashlib.sha256()
    h.update(candles["timestamps"].astype("int64").to_numpy().tobytes())
    for col in ["open", "high", "low", "close", "volume"]:
        h.update(candles[col].to_numpy(dtype="float64").tobytes())
    return h.hexdigest()[:16]


def config_key(config: dict) -> str:
    """Stable short hash of a JSON-serialisable config. Key order does not matter."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]


class ForecastCache:
    """One directory per configuration; one .npz file per forecast origin."""

    def __init__(self, root: Path, config: dict):
        self.dir = Path(root) / config_key(config)
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "config.json"
        if not meta.exists():
            meta.write_text(json.dumps(config, indent=2, sort_keys=True))

    def _path(self, origin: int) -> Path:
        return self.dir / f"origin_{origin}.npz"

    def has(self, origin: int) -> bool:
        return self._path(origin).exists()

    def put(self, origin: int, forecast: pd.DataFrame) -> None:
        """Write atomically: a crash mid-write leaves a .tmp file, never a half-written forecast."""
        tmp = self._path(origin).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez(
                f,
                values=forecast.to_numpy(),
                index=forecast.index.to_numpy().astype("datetime64[ns]"),
                columns=np.array(list(forecast.columns), dtype="U"),  # plain strings: loading never needs pickle
            )
        os.replace(tmp, self._path(origin))

    def get(self, origin: int) -> pd.DataFrame:
        with np.load(self._path(origin)) as z:
            return pd.DataFrame(z["values"], index=pd.DatetimeIndex(z["index"]), columns=list(z["columns"]))


def load_cached(windows: list[Window], cache: ForecastCache) -> dict[int, pd.DataFrame]:
    """Read forecasts for `windows` without any model. Raises if any are missing."""
    missing = [w.origin for w in windows if not cache.has(w.origin)]
    if missing:
        raise KeyError(f"{len(missing)} forecasts not cached, e.g. origins {missing[:5]}")
    return {w.origin: cache.get(w.origin) for w in windows}


def run_walk_forward(windows: list[Window], cache: ForecastCache, forecast_fn, progress=print) -> dict[int, pd.DataFrame]:
    """Forecast each window in chronological order, computing only what the cache lacks.

    `forecast_fn(window) -> DataFrame`. Results are always read back from the cache, so a fresh
    forecast and a cached one go through exactly the same path.
    """
    origins = [w.origin for w in windows]
    if origins != sorted(set(origins)):
        raise ValueError("windows must be unique and in chronological order")
    todo = [w for w in windows if not cache.has(w.origin)]
    progress(f"{len(windows) - len(todo)} cached, {len(todo)} to compute")
    start = time.time()
    for i, w in enumerate(todo, 1):
        cache.put(w.origin, forecast_fn(w))
        if i % 10 == 0 or i == len(todo):
            per = (time.time() - start) / i
            progress(f"  {i}/{len(todo)} done, {per:.1f}s each, ~{per * (len(todo) - i):.0f}s left")
    return load_cached(windows, cache)
