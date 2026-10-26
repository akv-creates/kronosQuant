import numpy as np
import pytest

from kronosquant.forecast import (
    FEATURES,
    forecast_window,
    load_predictor,
    load_sample_candles,
)

LOOKBACK, PRED_LEN = 400, 120


@pytest.fixture(scope="module")
def candles():
    return load_sample_candles()


@pytest.fixture(scope="module")
def predictor():
    return load_predictor("cpu")


@pytest.fixture(scope="module")
def pred(predictor, candles):
    return forecast_window(predictor, candles, len(candles) - PRED_LEN, LOOKBACK, PRED_LEN, seed=0)


def test_sample_data_is_sorted_and_clean(candles):
    assert candles["timestamps"].is_monotonic_increasing
    assert not candles[FEATURES].isna().any().any()


def test_forecast_shape_columns_and_index(pred, candles):
    expected_index = candles["timestamps"].iloc[-PRED_LEN:].reset_index(drop=True)
    assert pred.shape == (PRED_LEN, len(FEATURES))
    assert list(pred.columns) == FEATURES
    assert (pred.index.to_series().reset_index(drop=True) == expected_index).all()


def test_forecast_values_are_sane(pred):
    assert np.isfinite(pred.to_numpy()).all()
    assert (pred["high"] >= pred["low"]).all()  # observed for seed 0; the model samples candles, so this is not guaranteed in general


def test_same_seed_gives_identical_forecast_on_cpu(predictor, candles, pred):
    again = forecast_window(predictor, candles, len(candles) - PRED_LEN, LOOKBACK, PRED_LEN, seed=0)
    np.testing.assert_array_equal(pred.to_numpy(), again.to_numpy())


def test_window_must_fit_in_data(predictor, candles):
    with pytest.raises(ValueError):
        forecast_window(predictor, candles, 10, LOOKBACK, PRED_LEN)


def test_forecast_does_not_depend_on_future_values(predictor, candles, pred):
    """Leakage test: overwrite every feature at/after the origin with garbage. Only timestamps remain.

    If forecast_window leaked future values (e.g. an off-by-one slice), the forecast would change.
    """
    origin = len(candles) - PRED_LEN
    poisoned = candles.copy()
    poisoned.loc[origin:, FEATURES] = 1e9
    again = forecast_window(predictor, poisoned, origin, LOOKBACK, PRED_LEN, seed=0)
    np.testing.assert_array_equal(pred.to_numpy(), again.to_numpy())
