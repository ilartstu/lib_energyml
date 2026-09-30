"""Feature scaling learned on the training data."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from ..core.base import BaseTransformer, NotFittedError
from ..utils.validation import as_frame, numeric_columns

METHODS = ("standard", "minmax", "robust", "maxabs")


class ScalingProcessor(BaseTransformer):
    """``x' = (x - center) / scale`` per column, with parameters from ``fit``.

    * ``standard`` — mean / std
    * ``minmax`` — to ``feature_range`` (default 0..1)
    * ``robust`` — median / IQR
    * ``maxabs`` — divide by max |x| ("обезличивание"); ``scale={"POWER": 50}``
      divides by a fixed value instead, e.g. the installed capacity.

    >>> scaler = ScalingProcessor("minmax", columns=["T_AIR", "INSOLATION"])
    >>> train = scaler.fit_transform(train); test = scaler.transform(test)
    """

    stateless = False

    def __init__(self, method: str = "standard", columns: list[str] | None = None,
                 exclude: list[str] | None = None, feature_range: tuple[float, float] = (0.0, 1.0),
                 scale: dict[str, float] | None = None):
        self.method = method
        self.columns = columns
        self.exclude = exclude
        self.feature_range = feature_range
        self.scale = scale

    def _fit(self, df: pd.DataFrame) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.scale and self.method != "maxabs":
            raise ValueError("Fixed 'scale' values are only supported with method='maxabs'")
        cols = self.columns if self.columns is not None else numeric_columns(df)
        cols = [c for c in cols if c not in set(self.exclude or [])]
        center, scale = {}, {}
        for col in cols:
            x = pd.to_numeric(df[col], errors="coerce")
            x = x[np.isfinite(x)]
            if self.method == "standard":
                c, s = x.mean(), x.std(ddof=0)
            elif self.method == "minmax":
                c, s = x.min(), x.max() - x.min()
            elif self.method == "robust":
                c, s = x.median(), x.quantile(0.75) - x.quantile(0.25)
            else:
                c = 0.0
                s = (self.scale or {}).get(col, x.abs().max())
            if not np.isfinite(s) or s == 0:
                warnings.warn(f"Column {col!r} has zero spread; it is left unscaled", stacklevel=3)
                s = 1.0
            center[col], scale[col] = float(c), float(s)
        self.center_ = pd.Series(center, dtype=float)
        self.scale_ = pd.Series(scale, dtype=float)
        if self.method == "minmax":
            low, high = self.feature_range
            self.width_, self.offset_ = float(high - low), float(low)
        else:
            self.width_, self.offset_ = 1.0, 0.0

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in self.center_.index:
            if col in df.columns:
                df[col] = (df[col] - self.center_[col]) / self.scale_[col] * self.width_ + self.offset_
        self.report_ = {"method": self.method, "columns": list(self.center_.index)}
        return df

    def inverse_transform(self, data):
        if not getattr(self, "fitted_", False):
            raise NotFittedError("ScalingProcessor is not fitted")
        df = as_frame(data).copy()
        for col in self.center_.index:
            if col in df.columns:
                df[col] = (df[col] - self.offset_) / self.width_ * self.scale_[col] + self.center_[col]
        return df
