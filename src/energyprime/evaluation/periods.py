"""Metrics per window (day, 5 days...) and per group (hour of day, month).

This is what the notebooks call "rolling-origin" metrics: the test-period
predictions are cut into windows and every window is scored separately. It
shows how stable the quality is from day to day. (A real rolling-origin
backtest, where the model only knows data up to each origin, is done with
``Forecaster.predict_recursive`` / ``expanding_window_splits``.)
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pandas as pd

from ..utils.validation import infer_freq
from .metrics import evaluate_regression


def iter_windows(index: pd.DatetimeIndex, window: int | str, step: int | str | None = None,
                 start: str = "first") -> Iterator[tuple[int, int]]:
    """``(start_pos, end_pos_exclusive)`` of consecutive windows.

    ``window``/``step`` are row counts (``24``) or durations (``"24h"``).
    ``start="midnight"`` begins at the first 00:00 timestamp.
    """
    n = len(index)
    step = window if step is None else step
    first = 0
    if start == "midnight":
        midnights = np.flatnonzero((index.hour == 0) & (index.minute == 0))
        first = int(midnights[0]) if midnights.size else 0
    elif start != "first":
        raise ValueError("start must be 'first' or 'midnight'")
    if isinstance(window, int):
        s = first
        while s + window <= n:
            yield s, s + window
            s += step
        return
    window_td, step_td = pd.Timedelta(window), pd.Timedelta(step)
    t0 = index[first]
    data_end = index[-1] + (infer_freq(index) or pd.Timedelta(0))  # end of the last interval
    while t0 + window_td <= data_end:
        a = int(index.searchsorted(t0, side="left"))
        b = int(index.searchsorted(t0 + window_td, side="left"))
        if b > a:
            yield a, b
        t0 = t0 + step_td


def _series(values, index) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    return pd.Series(np.asarray(values, dtype=float).ravel(), index=index)


def evaluate_by_window(y_true, y_pred, window: int | str = 24, step: int | str | None = None,
                       metrics=("nmae", "nrmse", "r2"), *, norm: float | None = None, std=None,
                       lower=None, upper=None, start: str = "first", percent: bool = False) -> pd.DataFrame:
    """One row of metrics per window.

    >>> evaluate_by_window(y_test, pred, window=24, norm=50)          # per day
    >>> evaluate_by_window(y_test, pred, window=120, step=24, norm=50)  # 5-day windows, daily step
    """
    y_true = _series(y_true, getattr(y_pred, "index", None))
    index = y_true.index
    y_pred = _series(y_pred, index)
    extras = {k: (None if v is None else _series(v, index).to_numpy())
              for k, v in {"std": std, "lower": lower, "upper": upper}.items()}
    rows = []
    for a, b in iter_windows(index, window, step, start):
        sl = slice(a, b)
        scores = evaluate_regression(
            y_true.iloc[sl], y_pred.iloc[sl], metrics, norm=norm, percent=percent,
            **{k: (None if v is None else v[sl]) for k, v in extras.items()},
        )
        rows.append({"start": index[a], "end": index[b - 1], "n": b - a, **scores})
    return pd.DataFrame(rows)


def summarize_windows(window_metrics: pd.DataFrame, metrics=None) -> pd.DataFrame:
    """mean / std / min / max of window metrics (the notebooks' summary table)."""
    cols = metrics or [c for c in window_metrics.columns if c not in ("start", "end", "n")]
    return window_metrics[cols].agg(["mean", "std", "min", "max"])


def evaluate_by_group(y_true, y_pred, by: str | Callable = "hour", metrics=("nmae", "nrmse", "r2"), *,
                      norm: float | None = None, percent: bool = False) -> pd.DataFrame:
    """Metrics per hour of day / month / weekday (or any function of the index)."""
    y_true = _series(y_true, getattr(y_pred, "index", None))
    y_pred = _series(y_pred, y_true.index)
    keys = by(y_true.index) if callable(by) else getattr(y_true.index, by)
    rows = {}
    for key in pd.unique(np.asarray(keys)):
        mask = np.asarray(keys) == key
        rows[key] = {"n": int(mask.sum()),
                     **evaluate_regression(y_true[mask], y_pred[mask], metrics, norm=norm, percent=percent)}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()
