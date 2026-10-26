"""Stage 2: chronological splits and forecast windows that cannot see the future.

Everything here works on row positions (0..n-1) of a time-sorted DataFrame.
"""
from dataclasses import dataclass


def chronological_split(n_rows: int, train: float = 0.6, val: float = 0.2, embargo: int = 0) -> dict[str, range]:
    """Split rows into consecutive train/val/test ranges. Never shuffles.

    `embargo` rows are dropped after train and after val, so that nothing at the end of one period
    shares information with the start of the next (e.g. a window or feature that spans the boundary).
    """
    if not (0 < train and 0 < val and train + val < 1):
        raise ValueError("need train > 0, val > 0 and train + val < 1")
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    train_end = int(n_rows * train)
    val_start = train_end + embargo
    val_end = val_start + int(n_rows * val)
    test_start = val_end + embargo
    if test_start >= n_rows:
        raise ValueError("embargo too large for this much data")
    return {
        "train": range(0, train_end),
        "val": range(val_start, val_end),
        "test": range(test_start, n_rows),
    }


@dataclass(frozen=True)
class Window:
    """One forecast: look at `lookback` rows before `origin`, predict `pred_len` rows from `origin`."""

    origin: int
    lookback: int
    pred_len: int

    @property
    def history(self) -> range:
        return range(self.origin - self.lookback, self.origin)

    @property
    def future(self) -> range:
        return range(self.origin, self.origin + self.pred_len)


def make_windows(split: range, lookback: int, pred_len: int, step: int) -> list[Window]:
    """Forecast windows whose *future* lies entirely inside `split`.

    The history may reach back before the split start. That is fine for a frozen pretrained model
    (it is never fitted on that data) but would be leakage-in-reverse for anything fitted on it,
    so revisit this rule when we introduce fitted models.
    """
    if lookback < 1 or pred_len < 1 or step < 1:
        raise ValueError("lookback, pred_len and step must be >= 1")
    first = max(split.start, lookback)
    last = split.stop - pred_len  # origin + pred_len must not exceed split.stop
    return [Window(o, lookback, pred_len) for o in range(first, last + 1, step)]
