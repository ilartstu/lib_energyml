"""Joining several sources (station measurements + weather) on time."""
from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

import pandas as pd

from ..utils.validation import as_frame, check_time_index


def merge_by_time(sources: Mapping[str, object] | Sequence[object], how: str = "outer",
                  prefix: bool | None = None, sep: str = "_", drop_empty: bool = True,
                  duplicates: str = "first") -> pd.DataFrame:
    """Join DataFrames/Datasets on their DatetimeIndex.

    ``prefix``: ``True`` — prefix every column with its source label,
    ``None`` — only the column names that appear in several sources
    (``T_AIR`` -> ``station_T_AIR`` and ``nwp_T_AIR``), ``False`` — raise on
    collisions. ``duplicates``: which row to keep for repeated timestamps.

    >>> merge_by_time({"station": ds, "meteo": weather_df})
    """
    if not isinstance(sources, Mapping):
        sources = {str(i): s for i, s in enumerate(sources)}
    frames = {}
    for label, source in sources.items():
        frame = as_frame(source).copy()
        check_time_index(frame, f"source {label!r}")
        frame = frame[~frame.index.duplicated(keep=duplicates)]
        if drop_empty:
            frame = frame.dropna(axis=1, how="all")
        frames[label] = frame

    counts = Counter(c for f in frames.values() for c in f.columns)
    colliding = {c for c, n in counts.items() if n > 1}
    if colliding and prefix is False:
        raise ValueError(f"Column names repeat across sources: {sorted(colliding)}")
    renamed = []
    for label, frame in frames.items():
        if prefix:
            frame = frame.add_prefix(f"{label}{sep}")
        elif prefix is None:
            frame = frame.rename(columns={c: f"{label}{sep}{c}" for c in frame.columns if c in colliding})
        renamed.append(frame)
    return pd.concat(renamed, axis=1, join=how).sort_index()
