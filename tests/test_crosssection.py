import numpy as np
import pytest

from kronosquant.crosssection import rank_ic_by_date


def test_perfect_reversed_and_constant_rankings():
    actual = np.random.default_rng(0).normal(size=(5, 8))
    np.testing.assert_allclose(rank_ic_by_date(actual * 3 + 1, actual), 1.0)  # any increasing transform ranks the same
    np.testing.assert_allclose(rank_ic_by_date(-actual, actual), -1.0)
    out = rank_ic_by_date(np.zeros((5, 8)), actual)
    assert np.isnan(out).all()  # a constant forecast ranks nothing


def test_market_wide_move_does_not_matter():
    rng = np.random.default_rng(1)
    pred, actual = rng.normal(size=(40, 10)), rng.normal(size=(40, 10))
    shifted = actual + rng.normal(size=(40, 1)) * 100  # a big common move on each date, same for every asset
    np.testing.assert_allclose(rank_ic_by_date(pred, actual), rank_ic_by_date(pred, shifted))


def test_matches_hand_computed_spearman_and_ignores_asset_order():
    pred = np.array([[1.0, 2.0, 3.0, 4.0]])
    actual = np.array([[1.0, 3.0, 2.0, 4.0]])  # ranks differ by one swap: rho = 1 - 6*2/(4*15) = 0.8
    assert rank_ic_by_date(pred, actual)[0] == pytest.approx(0.8)
    perm = [2, 0, 3, 1]
    assert rank_ic_by_date(pred[:, perm], actual[:, perm])[0] == pytest.approx(0.8)


def test_unrelated_forecasts_average_near_zero_and_bad_shapes_rejected():
    rng = np.random.default_rng(2)
    ic = rank_ic_by_date(rng.normal(size=(2000, 12)), rng.normal(size=(2000, 12)))
    assert abs(np.nanmean(ic)) < 0.02
    assert ic.std() == pytest.approx(1 / np.sqrt(11), rel=0.1)  # sd of one date's IC is about 1/sqrt(assets-1)
    with pytest.raises(ValueError):
        rank_ic_by_date(np.zeros((3, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        rank_ic_by_date(np.zeros((3, 5)), np.zeros((4, 5)))
