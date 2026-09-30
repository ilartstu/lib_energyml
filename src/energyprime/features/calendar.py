"""Calendar features from the DatetimeIndex."""
from __future__ import annotations

import pandas as pd

from ..core.base import BaseTransformer
from ..utils.validation import check_time_index

CALENDAR = {
    "hour": lambda idx: idx.hour,
    "day": lambda idx: idx.day,
    "month": lambda idx: idx.month,
    "year": lambda idx: idx.year,
    "weekday": lambda idx: idx.weekday,
    "day_of_year": lambda idx: idx.dayofyear,
    "minute_of_day": lambda idx: idx.hour * 60 + idx.minute,
    "weekend": lambda idx: (idx.weekday >= 5).astype(int),
}


def calendar_values(index: pd.DatetimeIndex, name: str):
    if name not in CALENDAR:
        raise KeyError(f"Unknown calendar feature {name!r}; available: {sorted(CALENDAR)}")
    return CALENDAR[name](index)


class CalendarFeatures(BaseTransformer):
    """Add calendar columns (upper-case names: ``YEAR``, ``HOUR``, ``DAY_OF_YEAR``...).

    >>> CalendarFeatures(["year", "hour", "weekday"])
    """

    def __init__(self, features: list[str] = ("hour", "month", "day_of_year")):
        self.features = features

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_time_index(df)
        for name in self.features:
            df[name.upper()] = calendar_values(df.index, name)
        return df
