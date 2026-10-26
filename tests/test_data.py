import pandas as pd
import pytest

from kronosquant.data import DataError, find_gaps, validate_candles


def good():
    return pd.DataFrame(
        {
            "timestamps": pd.date_range("2024-01-01 09:35", periods=4, freq="5min"),
            "open": [10.0, 10.5, 10.4, 10.6],
            "high": [10.8, 10.9, 10.7, 10.9],
            "low": [9.9, 10.3, 10.2, 10.5],
            "close": [10.5, 10.4, 10.6, 10.8],
            "volume": [100, 120, 90, 0],
        }
    )


def test_good_data_passes():
    assert validate_candles(good()) is not None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.drop(columns="volume"), "missing columns"),
        (lambda d: d.assign(close=[10.5, None, 10.6, 10.8]), "NaN"),
        (lambda d: d.iloc[::-1].reset_index(drop=True), "not sorted"),
        (lambda d: pd.concat([d, d.iloc[[-1]]], ignore_index=True), "duplicate"),
        (lambda d: d.assign(low=[9.9, 0.0, 10.2, 10.5]), "non-positive"),
        (lambda d: d.assign(volume=[100, -1, 90, 0]), "negative volume"),
        (lambda d: d.assign(high=[10.8, 10.2, 10.7, 10.9]), "high is below"),
        (lambda d: d.assign(low=[9.9, 10.3, 10.7, 10.5]), "low is above"),
    ],
)
def test_bad_data_is_rejected(mutate, message):
    with pytest.raises(DataError, match=message):
        validate_candles(mutate(good()))


def test_find_gaps_reports_missing_bars():
    df = good()
    df.loc[2:, "timestamps"] += pd.Timedelta("1h")
    gaps = find_gaps(df)
    assert len(gaps) == 1
    assert gaps["gap"].iloc[0] == pd.Timedelta("1h05min")
