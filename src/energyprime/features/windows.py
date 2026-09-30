"""History as windows for sequence models (LSTM, GRU, transformers).

A window is a 3-D block ``(samples, lookback, features)``: for the moment
``t`` it holds the features of ``t-lookback+1 .. t``. Windows that cross a
time gap (deleted rows) or contain NaN are skipped, so the model never sees
hours glued together across missing days.

This is always the last step before the model: after it the data is a
NumPy array, not a DataFrame.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import Params
from ..utils.validation import infer_freq, to_timedelta


def make_windows(values: np.ndarray, lookback: int, index: pd.DatetimeIndex | None = None,
                 freq=None, check_gaps: bool = True,
                 ends: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build windows ending at every valid position.

    Returns ``X`` of shape ``(m, lookback, n_features)`` and the positions of
    the window ends (use them to pick the targets: ``y[positions]``).
    ``ends`` (bool per row) restricts which rows may end a window.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    n, n_features = values.shape
    lookback = int(lookback)
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    if n < lookback:
        return np.empty((0, lookback, n_features)), np.empty(0, dtype=int)

    t = np.arange(lookback - 1, n)
    finite_rows = np.isfinite(values).all(axis=1)
    bad = np.concatenate([[0], np.cumsum(~finite_rows)])
    ok = (bad[t + 1] - bad[t + 1 - lookback]) == 0

    if check_gaps and index is not None and lookback > 1:
        step = to_timedelta(freq) or infer_freq(index)
        if step is not None:
            steps = np.asarray(index[1:] - index[:-1])
            contiguous = np.concatenate([[False], steps == np.timedelta64(step)])
            breaks = np.concatenate([[0], np.cumsum(~contiguous)])
            ok &= (breaks[t + 1] - breaks[t + 2 - lookback]) == 0

    if ends is not None:
        ok &= np.asarray(ends, dtype=bool)[t]

    positions = t[ok]
    all_windows = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0)
    X = np.ascontiguousarray(all_windows[positions - (lookback - 1)].transpose(0, 2, 1))
    return X, positions


class WindowGenerator(Params):
    """Parameters of window building, kept with a fitted Forecaster.

    >>> X, positions = WindowGenerator(lookback=24).make(values, index)
    """

    def __init__(self, lookback: int = 24, freq: str | None = None, check_gaps: bool = True):
        self.lookback = lookback
        self.freq = freq
        self.check_gaps = check_gaps

    def make(self, values, index=None, ends=None) -> tuple[np.ndarray, np.ndarray]:
        return make_windows(values, self.lookback, index=index, freq=self.freq,
                            check_gaps=self.check_gaps, ends=ends)
