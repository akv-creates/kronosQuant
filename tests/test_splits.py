import pytest

from kronosquant.splits import chronological_split, make_windows


def test_split_is_ordered_disjoint_and_never_shuffled():
    s = chronological_split(1000, train=0.6, val=0.2)
    assert s["train"].stop <= s["val"].start
    assert s["val"].stop <= s["test"].start
    assert s["test"].stop == 1000
    assert len(s["train"]) == 600 and len(s["val"]) == 200 and len(s["test"]) == 200


def test_embargo_leaves_a_gap_between_periods():
    s = chronological_split(1000, 0.6, 0.2, embargo=10)
    assert s["val"].start - s["train"].stop == 10
    assert s["test"].start - s["val"].stop == 10


@pytest.mark.parametrize("kwargs", [dict(train=0.9, val=0.2), dict(train=0.0, val=0.5), dict(embargo=-1)])
def test_split_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        chronological_split(1000, **kwargs)


def test_windows_never_look_past_their_origin():
    for w in make_windows(range(0, 1000), lookback=50, pred_len=10, step=7):
        assert max(w.history) < w.origin == min(w.future)  # history strictly before future
        assert len(w.history) == 50 and len(w.future) == 10


def test_window_futures_stay_inside_the_split():
    split = range(300, 500)
    windows = make_windows(split, lookback=50, pred_len=10, step=5)
    assert windows
    for w in windows:
        assert w.future.start >= split.start and w.future.stop <= split.stop


def test_last_window_ends_exactly_at_split_end_and_none_fit_when_too_short():
    ws = make_windows(range(100, 200), lookback=50, pred_len=10, step=1)
    assert ws[-1].future.stop == 200
    assert make_windows(range(100, 105), lookback=50, pred_len=10, step=1) == []


def test_history_may_reach_before_split_but_never_before_row_zero():
    ws = make_windows(range(0, 100), lookback=40, pred_len=10, step=1)
    assert ws[0].origin == 40 and ws[0].history.start == 0


def test_real_sample_data_splits_are_chronological_and_windows_stay_in_period():
    from kronosquant.forecast import load_sample_candles

    candles = load_sample_candles()
    ts = candles["timestamps"]
    s = chronological_split(len(candles), embargo=120)
    assert ts.iloc[s["train"][-1]] < ts.iloc[s["val"][0]] < ts.iloc[s["val"][-1]] < ts.iloc[s["test"][0]]
    for w in make_windows(s["test"], lookback=400, pred_len=120, step=500):
        assert ts.iloc[w.future[0]] > ts.iloc[s["val"][-1]]  # test forecasts start after validation ends
