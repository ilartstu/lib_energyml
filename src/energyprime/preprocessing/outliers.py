"""Outliers: detectors (which points look wrong) and actions (what to do with them).

Detectors from the SVGP notebook and the Figures app:

* ``zscore`` — ``|x - mean| / std > threshold`` (statistics from training data)
* ``iqr`` — outside ``[Q1 - factor*IQR, Q3 + factor*IQR]``
* ``hampel`` — deviation from a centred rolling median in units of 1.4826*MAD
* ``rolling_median`` — deviation from a rolling median in units of rolling std
* ``diff`` — jump between neighbours larger than ``threshold * std(diff)``
* ``physical`` — outside the physical ``limits`` of the column (always applied)

Missing values (NaN and, where configured, zeros) are never outliers and are
excluded from the statistics.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..core.base import BaseTransformer, Params
from ..core.registry import Registry
from ..utils.validation import as_frame, numeric_columns
from .missing import interpolate_positions

DETECTORS = Registry("outlier detector")
ACTIONS = ("interpolate", "set_nan", "clip", "drop_rows", "flag")


def _values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values[np.isinf(values)] = np.nan
    return values


def _ignore(values: np.ndarray, ignore: np.ndarray | None) -> np.ndarray:
    mask = ~np.isfinite(values)
    if ignore is not None:
        mask |= np.asarray(ignore, dtype=bool)
    return mask


class Detector(Params):
    stateful: ClassVar[bool] = False

    def fit(self, series: pd.Series, ignore: np.ndarray | None = None) -> "Detector":
        return self

    def detect(self, series: pd.Series, ignore: np.ndarray | None = None) -> np.ndarray:
        values = _values(series)
        skip = _ignore(values, ignore)
        with np.errstate(invalid="ignore"):
            flags = self._flags(values, skip)
        return np.asarray(flags, dtype=bool) & ~skip

    def bounds(self) -> tuple[float | None, float | None] | None:
        """Global (low, high) bounds if the detector has them (used by ``clip``)."""
        return None

    def _flags(self, values: np.ndarray, skip: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def _stats_values(values: np.ndarray, skip: np.ndarray, exclude_zeros: bool) -> np.ndarray:
    keep = ~skip
    if exclude_zeros:
        keep &= values != 0
    return values[keep]


@DETECTORS.register("zscore")
class ZScoreDetector(Detector):
    stateful = True

    def __init__(self, threshold: float = 3.0, exclude_zeros: bool = True, min_points: int = 10, ddof: int = 1):
        self.threshold = threshold
        self.exclude_zeros = exclude_zeros
        self.min_points = min_points
        self.ddof = ddof

    def fit(self, series, ignore=None):
        values = _values(series)
        clean = _stats_values(values, _ignore(values, ignore), self.exclude_zeros)
        self.mean_ = self.std_ = None
        if clean.size >= self.min_points:
            std = float(np.std(clean, ddof=self.ddof))
            if std > 0 and np.isfinite(std):
                self.mean_, self.std_ = float(np.mean(clean)), std
        return self

    def _flags(self, values, skip):
        if getattr(self, "std_", None) is None:
            return np.zeros(values.size, dtype=bool)
        return np.abs(values - self.mean_) / self.std_ > self.threshold

    def bounds(self):
        if getattr(self, "std_", None) is None:
            return None
        return (self.mean_ - self.threshold * self.std_, self.mean_ + self.threshold * self.std_)


@DETECTORS.register("iqr")
class IQRDetector(Detector):
    stateful = True

    def __init__(self, factor: float = 1.5, exclude_zeros: bool = True, min_points: int = 10):
        self.factor = factor
        self.exclude_zeros = exclude_zeros
        self.min_points = min_points

    def fit(self, series, ignore=None):
        values = _values(series)
        clean = _stats_values(values, _ignore(values, ignore), self.exclude_zeros)
        self.low_ = self.high_ = None
        if clean.size >= self.min_points:
            q1, q3 = np.quantile(clean, [0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                self.low_, self.high_ = float(q1 - self.factor * iqr), float(q3 + self.factor * iqr)
        return self

    def _flags(self, values, skip):
        if getattr(self, "low_", None) is None:
            return np.zeros(values.size, dtype=bool)
        return (values < self.low_) | (values > self.high_)

    def bounds(self):
        return None if getattr(self, "low_", None) is None else (self.low_, self.high_)


@DETECTORS.register("hampel")
class HampelDetector(Detector):
    def __init__(self, window: int = 72, n_sigmas: float = 3.0, exclude_zeros: bool = True,
                 min_periods: int | None = None):
        self.window = window
        self.n_sigmas = n_sigmas
        self.exclude_zeros = exclude_zeros
        self.min_periods = min_periods

    def _flags(self, values, skip):
        stats = values.copy()
        stats[skip] = np.nan
        if self.exclude_zeros:
            stats[stats == 0] = np.nan
        s = pd.Series(stats)
        min_periods = self.min_periods or max(10, self.window // 3)
        median = s.rolling(self.window, center=True, min_periods=min_periods).median()
        deviation = (s - median).abs()
        mad = deviation.rolling(self.window, center=True, min_periods=min_periods).median()
        sigma = 1.4826 * mad
        flags = (deviation > self.n_sigmas * sigma) & sigma.notna() & (sigma > 0)
        return flags.to_numpy(dtype=bool)


@DETECTORS.register("rolling_median")
class RollingMedianDetector(Detector):
    def __init__(self, window: int = 24, threshold: float = 3.0):
        self.window = window
        self.threshold = threshold

    def _flags(self, values, skip):
        s = pd.Series(np.where(skip, np.nan, values))
        min_periods = max(3, self.window // 3)
        median = s.rolling(self.window, center=True, min_periods=min_periods).median()
        std = s.rolling(self.window, center=True, min_periods=min_periods).std()
        deviation = (s - median).abs() / std.replace(0, np.nan)
        return (deviation > self.threshold).to_numpy(dtype=bool, na_value=False)


@DETECTORS.register("diff")
class DiffDetector(Detector):
    stateful = True

    def __init__(self, threshold: float = 4.0, abs_threshold: float | None = None):
        self.threshold = threshold
        self.abs_threshold = abs_threshold

    @staticmethod
    def _diff(values: np.ndarray) -> np.ndarray:
        d = np.diff(values, prepend=values[:1])
        d[~np.isfinite(d)] = 0.0
        return d

    def fit(self, series, ignore=None):
        d = self._diff(np.where(_ignore(_values(series), ignore), np.nan, _values(series)))
        nonzero = d[d != 0]
        self.std_ = float(np.std(nonzero)) if nonzero.size else 0.0
        return self

    def _flags(self, values, skip):
        d = np.abs(self._diff(np.where(skip, np.nan, values)))
        if self.abs_threshold is not None:
            return d > float(self.abs_threshold)
        std = getattr(self, "std_", 0.0)
        return d > self.threshold * std if std else np.zeros(values.size, dtype=bool)


@DETECTORS.register("physical")
class PhysicalLimitsDetector(Detector):
    def __init__(self, lower: float | None = None, upper: float | None = None):
        self.lower = lower
        self.upper = upper

    def _flags(self, values, skip):
        flags = np.zeros(values.size, dtype=bool)
        if self.lower is not None:
            flags |= values < self.lower
        if self.upper is not None:
            flags |= values > self.upper
        return flags

    def bounds(self):
        return (self.lower, self.upper)


def make_detector(name: str, params: dict | None = None) -> Detector:
    return DETECTORS.get(name)(**(params or {}))


class OutlierProcessor(BaseTransformer):
    """Detect outliers per column and apply an action to the selected ones.

    >>> OutlierProcessor(
    ...     detectors={"zscore": {"threshold": 5}, "iqr": {"factor": 4}, "hampel": {"window": 72}},
    ...     apply=["zscore"],                    # the others only appear in diagnose()
    ...     limits={"T_AIR": (-50, 60)},         # physical limits are always applied
    ...     action="interpolate",
    ... )

    ``ignore_zeros``: zeros are missing values, not outliers (True / list of columns).
    """

    stateless = False

    def __init__(self, detectors: dict[str, dict] | None = None, apply: list[str] | None = None,
                 limits: dict[str, Any] | None = None, columns: list[str] | None = None,
                 action: str = "interpolate", ignore_zeros: bool | list[str] = False,
                 method: str = "linear"):
        self.detectors = detectors
        self.apply = apply
        self.limits = limits
        self.columns = columns
        self.action = action
        self.ignore_zeros = ignore_zeros
        self.method = method

    @property
    def changes_rows(self) -> bool:  # type: ignore[override]
        return self.action == "drop_rows"

    def _ignore_mask(self, df: pd.DataFrame, col: str) -> np.ndarray:
        values = _values(df[col])
        mask = ~np.isfinite(values)
        zeros = self.ignore_zeros is True or (isinstance(self.ignore_zeros, (list, tuple))
                                              and col in self.ignore_zeros)
        if zeros:
            mask |= values == 0
        return mask

    def _fit(self, df: pd.DataFrame) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}, got {self.action!r}")
        detectors = self.detectors if self.detectors is not None else {"zscore": {}}
        apply = list(detectors) if self.apply is None else list(self.apply)
        unknown = set(apply) - set(detectors)
        if unknown:
            raise ValueError(f"apply={sorted(unknown)} is not in detectors={list(detectors)}")
        self.apply_ = apply
        self.columns_ = self.columns if self.columns is not None else numeric_columns(df)
        limits = {c: tuple(v) for c, v in (self.limits or {}).items()}
        self.fitted_detectors_: dict[str, dict[str, Detector]] = {}
        for col in self.columns_:
            if col not in df.columns:
                continue
            ignore = self._ignore_mask(df, col)
            per_col = {name: make_detector(name, params).fit(df[col], ignore)
                       for name, params in detectors.items()}
            if col in limits and limits[col] != (None, None):
                per_col["physical"] = PhysicalLimitsDetector(*limits[col])
            self.fitted_detectors_[col] = per_col

    def detect(self, data) -> dict[str, pd.DataFrame]:
        """Masks of every detector plus ``"applied"`` (what the action will touch)."""
        df = as_frame(data)
        masks: dict[str, dict[str, np.ndarray]] = {}
        applied = {}
        for col, per_col in self.fitted_detectors_.items():
            if col not in df.columns:
                continue
            ignore = self._ignore_mask(df, col)
            final = np.zeros(len(df), dtype=bool)
            for name, detector in per_col.items():
                flags = detector.detect(df[col], ignore)
                masks.setdefault(name, {})[col] = flags
                if name in self.apply_ or name == "physical":
                    final |= flags
            applied[col] = final
        out = {name: pd.DataFrame(cols, index=df.index).reindex(columns=list(applied), fill_value=False)
               for name, cols in masks.items()}
        out["applied"] = pd.DataFrame(applied, index=df.index)
        return out

    def diagnose(self, data) -> pd.DataFrame:
        """Counts and shares per column for every detector and for the applied set."""
        return self.diagnose_from(self.detect(data))

    def _bounds(self, col: str) -> tuple[float, float]:
        low, high = -np.inf, np.inf
        for name, detector in self.fitted_detectors_[col].items():
            if name not in self.apply_ and name != "physical":
                continue
            b = detector.bounds()
            if b is None:
                continue
            if b[0] is not None:
                low = max(low, b[0])
            if b[1] is not None:
                high = min(high, b[1])
        return low, high

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        masks = self.detect(df)
        applied = masks["applied"]
        if self.action == "drop_rows":
            df = df.loc[~applied.any(axis=1).to_numpy()]
        else:
            for col in applied.columns:
                flags = applied[col].to_numpy(dtype=bool)
                if not flags.any():
                    continue
                values = _values(df[col])
                if self.action == "interpolate":
                    index = df.index if self.method == "time" else None
                    values, _ = interpolate_positions(values, flags, self.method, index)
                elif self.action == "set_nan":
                    values[flags] = np.nan
                elif self.action == "clip":
                    low, high = self._bounds(col)
                    clipped = np.clip(values, low, high)
                    no_bounds = flags & (clipped == values)  # detector without global bounds
                    values = np.where(flags, clipped, values)
                    values[no_bounds] = np.nan
                elif self.action == "flag":
                    df[f"{col}_outlier"] = flags
                    continue
                df[col] = values
        self.report_ = {"summary": self.diagnose_from(masks), "mask": applied}
        return df

    @staticmethod
    def diagnose_from(masks: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows = {}
        for name, mask in masks.items():
            rows[name] = mask.sum()
            rows[f"{name}_pct"] = (mask.mean() * 100).round(4)
        return pd.DataFrame(rows)
