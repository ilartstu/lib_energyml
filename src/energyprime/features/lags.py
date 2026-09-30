"""History as extra columns for table models (XGBoost, SVGP, linear...).

Lags are taken by time, not by row: after deleted rows a "lag 1" is NaN
instead of silently being a value from days ago.
"""
from __future__ import annotations

import pandas as pd

from ..core.base import BaseTransformer
from ..utils.validation import check_time_index, infer_freq, to_timedelta


def _freq(df: pd.DataFrame, freq) -> pd.Timedelta:
    check_time_index(df)
    step = to_timedelta(freq) or infer_freq(df.index)
    if step is None:
        raise ValueError("Cannot infer the time step; pass freq='1h'")
    return step


class LagFeatures(BaseTransformer):
    """``{col}_lag_{k}`` = value ``k`` steps earlier.

    >>> LagFeatures(["POWER"], lags=[1, 24, 48, 120])
    """

    def __init__(self, columns: list[str], lags: list[int], freq: str | None = None):
        self.columns = columns
        self.lags = lags
        self.freq = freq

    def max_steps(self) -> int:
        return max(self.lags) if self.lags else 0

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        step = _freq(df, self.freq)
        new = {}
        for col in self.columns:
            for k in self.lags:
                # move timestamps k steps forward, then align: value at t comes from t - k*step
                new[f"{col}_lag_{k}"] = df[col].shift(k, freq=step).reindex(df.index)
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


class RollingFeatures(BaseTransformer):
    """``{col}_roll_{stat}_{w}`` over the previous ``w`` steps (current value excluded).

    Excluding the current value makes rolling statistics of the target safe
    to use (no leakage).

    >>> RollingFeatures(["POWER"], windows=[24], stats=["mean", "std"])
    """

    def __init__(self, columns: list[str], windows: list[int], stats: list[str] = ("mean", "std"),
                 freq: str | None = None, include_current: bool = False, ddof: int = 1):
        self.columns = columns
        self.windows = windows
        self.stats = stats
        self.freq = freq
        self.include_current = include_current
        self.ddof = ddof

    def max_steps(self) -> int:
        return max(self.windows) if self.windows else 0

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        step = _freq(df, self.freq)
        closed = "right" if self.include_current else "left"
        new = {}
        for col in self.columns:
            for w in self.windows:
                roll = df[col].rolling(w * step, closed=closed, min_periods=w)
                for stat in self.stats:
                    if stat == "std":
                        values = roll.std(ddof=self.ddof)
                    elif stat == "var":
                        values = roll.var(ddof=self.ddof)
                    else:
                        values = getattr(roll, stat)()
                    new[f"{col}_roll_{stat}_{w}"] = values
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
