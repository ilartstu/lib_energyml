"""Time axis: order, duplicate timestamps, regular grid, resampling."""
from __future__ import annotations

import pandas as pd

from ..core.base import BaseTransformer
from ..utils.validation import check_time_index, infer_freq, to_timedelta

DUPLICATES = ("mean", "first", "last", "drop", "keep", "error")


class TimeProcessor(BaseTransformer):
    """Sort by time, resolve duplicate timestamps, optionally resample.

    ``duplicates``: ``mean`` (average numeric values), ``first``, ``last``,
    ``drop`` (remove all copies), ``keep``, ``error``.
    ``fill_missing_timestamps=True`` inserts absent timestamps of the regular
    grid as NaN rows, so that gaps become visible to the missing-value rules.

    >>> TimeProcessor(duplicates="mean", freq="1h", fill_missing_timestamps=True)
    """

    changes_rows = True

    def __init__(self, duplicates: str = "mean", freq: str | None = None,
                 fill_missing_timestamps: bool = False, resample: str | None = None, agg: str = "mean"):
        self.duplicates = duplicates
        self.freq = freq
        self.fill_missing_timestamps = fill_missing_timestamps
        self.resample = resample
        self.agg = agg

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_time_index(df)
        if self.duplicates not in DUPLICATES:
            raise ValueError(f"duplicates must be one of {DUPLICATES}")
        rows_in = len(df)
        was_sorted = bool(df.index.is_monotonic_increasing)
        if not was_sorted:
            df = df.sort_index(kind="stable")

        dup_mask = df.index.duplicated(keep=False)
        n_dup_timestamps = int(df.index[dup_mask].nunique())
        if n_dup_timestamps:
            if self.duplicates == "error":
                raise ValueError(f"{n_dup_timestamps} duplicated timestamps")
            if self.duplicates == "mean":
                agg = {c: ("mean" if pd.api.types.is_numeric_dtype(df[c]) else "first") for c in df.columns}
                df = df.groupby(level=0, sort=True).agg(agg)
            elif self.duplicates in ("first", "last"):
                df = df[~df.index.duplicated(keep=self.duplicates)]
            elif self.duplicates == "drop":
                df = df[~dup_mask]

        if self.resample:
            numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            df = df[numeric].resample(self.resample).agg(self.agg)

        freq = to_timedelta(self.resample or self.freq) or infer_freq(df.index)
        missing_timestamps = inserted = 0
        if freq is not None and len(df) > 1 and df.index.is_unique:
            grid = pd.date_range(df.index.min(), df.index.max(), freq=freq, tz=df.index.tz)
            absent = grid.difference(df.index)
            missing_timestamps = len(absent)
            if self.fill_missing_timestamps and missing_timestamps:
                name = df.index.name
                df = df.reindex(df.index.union(grid))
                df.index.name = name
                inserted = missing_timestamps

        self.report_ = {
            "rows_in": rows_in,
            "rows_out": len(df),
            "was_sorted": was_sorted,
            "duplicate_timestamps": n_dup_timestamps,
            "freq": str(freq) if freq is not None else None,
            "missing_timestamps": missing_timestamps,
            "inserted_rows": inserted,
        }
        return df
