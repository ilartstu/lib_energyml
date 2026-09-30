"""Sin/cos encoding of cyclic quantities (hour, day of year, wind direction)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.base import BaseTransformer
from ..utils.validation import check_time_index
from .calendar import CALENDAR, calendar_values


class CyclicEncoder(BaseTransformer):
    """Replace a cyclic value by ``sin(2*pi*x/period)`` and ``cos(2*pi*x/period)``.

    Keys are column names (``"DIRECTION": 360``) or calendar quantities taken
    from the index (``"hour": 24``, ``"day_of_year": 365.25``, ``"month": 12``).
    Names follow the notebooks: ``HOUR_SIN``, ``DAY_OF_YEAR_COS``, ``DIRECTION_SIN``.

    >>> CyclicEncoder({"hour": 24, "day_of_year": 365.25, "DIRECTION": 360})
    """

    def __init__(self, periods: dict[str, float], drop_original: bool = True,
                 suffixes: tuple[str, str] = ("_SIN", "_COS")):
        self.periods = periods
        self.drop_original = drop_original
        self.suffixes = suffixes

    def output_names(self, name: str) -> tuple[str, str]:
        base = name.upper() if name in CALENDAR else name
        return base + self.suffixes[0], base + self.suffixes[1]

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        to_drop = []
        for name, period in self.periods.items():
            if name in df.columns:
                values = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
                if self.drop_original:
                    to_drop.append(name)
            elif name in CALENDAR:
                check_time_index(df)
                values = np.asarray(calendar_values(df.index, name), dtype=float)
            else:
                raise KeyError(f"{name!r} is neither a column nor a calendar quantity ({sorted(CALENDAR)})")
            sin_name, cos_name = self.output_names(name)
            angle = 2 * np.pi * values / float(period)
            df[sin_name] = np.sin(angle)
            df[cos_name] = np.cos(angle)
        return df.drop(columns=to_drop)
