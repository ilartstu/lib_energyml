"""Small input checks shared by all modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def as_frame(data) -> pd.DataFrame:
    """Accept a DataFrame or a Dataset (anything with a ``.data`` DataFrame)."""
    if isinstance(data, pd.DataFrame):
        return data
    inner = getattr(data, "data", None)
    if isinstance(inner, pd.DataFrame):
        return inner
    raise TypeError(f"Expected a pandas DataFrame or a Dataset, got {type(data).__name__}")


def check_time_index(df: pd.DataFrame, name: str = "data") -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"{name} must have a DatetimeIndex (load it with energyprime.datasets "
            f"or call df.set_index(<time column>) first)"
        )


def infer_freq(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    """Most common step between consecutive timestamps (robust to gaps)."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return None
    diffs = pd.Series(index[1:] - index[:-1])
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return diffs.mode().iloc[0]


def to_timedelta(freq) -> pd.Timedelta | None:
    if freq is None:
        return None
    if isinstance(freq, pd.Timedelta):
        return freq
    return pd.to_timedelta(pd.tseries.frequencies.to_offset(freq))


def numeric_columns(df: pd.DataFrame, columns=None) -> list[str]:
    cols = list(df.columns) if columns is None else [c for c in columns if c in df.columns]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()
