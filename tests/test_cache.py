import numpy as np
import pandas as pd
import pytest

from kronosquant.cache import ForecastCache, config_key, data_fingerprint, load_cached, run_walk_forward
from kronosquant.splits import Window


def fake_forecast(origin: int, h: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=h, freq="5min")
    return pd.DataFrame(np.arange(h * 2, dtype="float32").reshape(h, 2) + origin, index=idx, columns=["close", "volume"])


def windows(origins):
    return [Window(o, 10, 3) for o in origins]


CFG = {"model": "x", "T": 1.0}


def test_roundtrip_is_exact_including_dtype_index_and_columns(tmp_path):
    cache = ForecastCache(tmp_path, CFG)
    original = fake_forecast(7)
    cache.put(7, original)
    pd.testing.assert_frame_equal(cache.get(7), original, check_freq=False)
    assert cache.get(7).to_numpy().dtype == np.float32


def test_second_run_computes_nothing_and_resume_computes_only_missing(tmp_path):
    cache = ForecastCache(tmp_path, CFG)
    calls = []

    def fn(w):
        calls.append(w.origin)
        return fake_forecast(w.origin)

    run_walk_forward(windows([1, 2, 3]), cache, fn, progress=lambda m: None)
    assert calls == [1, 2, 3]
    run_walk_forward(windows([1, 2, 3]), cache, fn, progress=lambda m: None)
    assert calls == [1, 2, 3]  # nothing recomputed
    out = run_walk_forward(windows([1, 2, 3, 4]), cache, fn, progress=lambda m: None)
    assert calls == [1, 2, 3, 4] and sorted(out) == [1, 2, 3, 4]  # only the new window was computed


def test_crash_midway_keeps_finished_work_and_leaves_no_partial_files(tmp_path):
    cache = ForecastCache(tmp_path, CFG)

    def flaky(w):
        if w.origin == 3:
            raise RuntimeError("boom")
        return fake_forecast(w.origin)

    with pytest.raises(RuntimeError):
        run_walk_forward(windows([1, 2, 3]), cache, flaky, progress=lambda m: None)
    assert cache.has(1) and cache.has(2) and not cache.has(3)
    assert not list(cache.dir.glob("*.tmp"))


def test_different_config_gets_a_different_directory(tmp_path):
    a, b = ForecastCache(tmp_path, CFG), ForecastCache(tmp_path, {**CFG, "T": 0.8})
    a.put(1, fake_forecast(1))
    assert a.dir != b.dir and not b.has(1)


def test_config_key_ignores_dict_order_but_not_values():
    assert config_key({"a": 1, "b": 2}) == config_key({"b": 2, "a": 1})
    assert config_key({"a": 1}) != config_key({"a": 2})


def test_windows_must_be_chronological_and_unique(tmp_path):
    cache = ForecastCache(tmp_path, CFG)
    for bad in ([3, 2, 1], [1, 1, 2]):
        with pytest.raises(ValueError):
            run_walk_forward(windows(bad), cache, lambda w: fake_forecast(w.origin), progress=lambda m: None)


def test_load_cached_reports_missing_and_never_needs_a_model(tmp_path):
    cache = ForecastCache(tmp_path, CFG)
    cache.put(1, fake_forecast(1))
    assert list(load_cached(windows([1]), cache)) == [1]
    with pytest.raises(KeyError, match="1 forecasts not cached"):
        load_cached(windows([1, 2]), cache)


def test_data_fingerprint_changes_when_any_value_changes():
    df = pd.DataFrame(
        {
            "timestamps": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0,
        }
    )
    base = data_fingerprint(df)
    assert data_fingerprint(df.copy()) == base
    changed = df.copy()
    changed.loc[4, "close"] = 1.5000001
    assert data_fingerprint(changed) != base
    assert data_fingerprint(df.iloc[:4]) != base


def test_forecast_config_changes_with_every_setting():
    from kronosquant.forecast import forecast_config, load_sample_candles

    c = load_sample_candles()
    base = forecast_config(c, "cpu", 400, 12, 1.0, 0.9, 5)
    for variant in [
        forecast_config(c, "mps", 400, 12, 1.0, 0.9, 5),
        forecast_config(c, "cpu", 300, 12, 1.0, 0.9, 5),
        forecast_config(c, "cpu", 400, 24, 1.0, 0.9, 5),
        forecast_config(c, "cpu", 400, 12, 0.8, 0.9, 5),
        forecast_config(c, "cpu", 400, 12, 1.0, 0.95, 5),
        forecast_config(c, "cpu", 400, 12, 1.0, 0.9, 1),
        forecast_config(c.iloc[:-1], "cpu", 400, 12, 1.0, 0.9, 5),
    ]:
        assert config_key(variant) != config_key(base)
