"""Dataset = time-indexed table + station specification."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from ..core.schema import StationSpec
from ..utils.validation import check_time_index, infer_freq, to_timedelta


@dataclass
class Dataset:
    """Measurements of one station with a sorted DatetimeIndex.

    ``data`` keeps every column of the source file; ``spec`` says which of them
    is the target, which are features and how to clean them. ``info`` is the
    loading report (unparsed timestamps, values that were not numbers...).
    """

    data: pd.DataFrame
    spec: StationSpec = field(default_factory=StationSpec)
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data, pd.DataFrame):
            raise TypeError(f"data must be a DataFrame, got {type(self.data).__name__}")
        check_time_index(self.data, "Dataset.data")

    # ---------------------------------------------------------------- columns
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def target(self) -> str | None:
        return self.spec.target

    @property
    def features(self) -> list[str]:
        return [c for c in self.spec.features if c in self.data.columns]

    @property
    def y(self) -> pd.Series:
        if self.target is None:
            raise ValueError("The station spec has no target column")
        return self.data[self.target]

    @property
    def X(self) -> pd.DataFrame:
        return self.data[self.features]

    @property
    def capacity(self) -> float | None:
        return self.spec.capacity

    @property
    def freq(self) -> pd.Timedelta | None:
        return to_timedelta(self.spec.time.freq) or infer_freq(self.data.index)

    def reference_power(self) -> tuple[float, str]:
        """Value to normalise errors by: capacity if known, else the target maximum."""
        if self.capacity is not None:
            return float(self.capacity), "capacity"
        return float(self.y.abs().max()), "max"

    # ------------------------------------------------------------- selection
    def with_data(self, data: pd.DataFrame) -> "Dataset":
        return Dataset(data, self.spec, copy.deepcopy(self.info))

    def slice(self, start=None, end=None) -> "Dataset":
        """Rows with ``start <= time <= end`` (strings like "2020-05" work)."""
        return self.with_data(self.data.loc[start:end])

    def copy(self) -> "Dataset":
        return Dataset(self.data.copy(), self.spec.copy(), copy.deepcopy(self.info))

    def head(self, n: int = 5) -> pd.DataFrame:
        return self.data.head(n)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        if len(self.data):
            period = f"{self.data.index.min()} -> {self.data.index.max()}"
        else:
            period = "empty"
        return (f"Dataset({self.name!r}, rows={len(self.data)}, period={period}, "
                f"target={self.target!r}, features={len(self.features)})")
