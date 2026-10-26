"""Stage 3: simple forecasters Kronos must beat. Each uses only the history it is given.

A forecast here is a path of cumulative log-returns from the last observed close:
element h-1 is the predicted log(close at step h / last close), for h = 1..horizon.
"""
import numpy as np


def random_walk(history_close: np.ndarray, horizon: int) -> np.ndarray:
    """Best guess is 'nothing changes': predicted return is zero at every step."""
    return np.zeros(horizon)


def drift(history_close: np.ndarray, horizon: int) -> np.ndarray:
    """Keep adding the average per-candle log-return seen in the history."""
    mean_step = np.diff(np.log(history_close)).mean()
    return mean_step * np.arange(1, horizon + 1)


def momentum(history_close: np.ndarray, horizon: int) -> np.ndarray:
    """Whatever happened over the last `horizon` candles keeps happening at the same pace."""
    past = np.log(history_close[-1] / history_close[-1 - horizon])
    return past * np.arange(1, horizon + 1) / horizon


BASELINES = {"random_walk": random_walk, "drift": drift, "momentum": momentum}
