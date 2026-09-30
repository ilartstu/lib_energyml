"""Target transforms with an exact inverse (applied inside the Forecaster)."""
from __future__ import annotations

import numpy as np
from scipy.special import expit

from ..core.base import NotFittedError, Params
from ..core.registry import Registry

TARGET_TRANSFORMS = Registry("target transform")


class TargetTransform(Params):
    """``fit(y)``, ``transform(y)``, ``inverse_transform(z)`` on 1-D arrays.

    Transforms are monotonic, so an interval ``mean ± k*std`` in the model's
    space maps to a valid interval in physical units.
    """

    def fit(self, y) -> "TargetTransform":
        return self

    def transform(self, y) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def inverse_transform(self, z) -> np.ndarray:
        return np.asarray(z, dtype=float)

    def fit_transform(self, y) -> np.ndarray:
        return self.fit(y).transform(y)

    @staticmethod
    def _finite(y) -> np.ndarray:
        y = np.asarray(y, dtype=float).ravel()
        return y[np.isfinite(y)]

    def _check(self) -> None:
        if not hasattr(self, "min_"):
            raise NotFittedError(f"{type(self).__name__} is not fitted")


@TARGET_TRANSFORMS.register("identity")
class IdentityTarget(TargetTransform):
    """No transformation."""


@TARGET_TRANSFORMS.register("minmax")
class MinMaxTarget(TargetTransform):
    """``(y - min) / (max - min)`` with min/max from the training target (LSTM notebook)."""

    def __init__(self, eps: float = 0.0):
        self.eps = eps

    def fit(self, y):
        y = self._finite(y)
        self.min_ = float(y.min())
        scale = float(y.max() - y.min() + self.eps)
        self.scale_ = scale if scale > 0 else 1.0
        return self

    def transform(self, y):
        self._check()
        return (np.asarray(y, dtype=float) - self.min_) / self.scale_

    def inverse_transform(self, z):
        self._check()
        return np.asarray(z, dtype=float) * self.scale_ + self.min_


@TARGET_TRANSFORMS.register("minmax_logit")
class MinMaxLogitTarget(TargetTransform):
    """Min-max to (0, 1), then logit — keeps predictions inside the observed range (SVGP notebook).

    ``z = clip(logit(clip((y - min) / (max - min + denom_eps), eps, 1 - eps)), -clip, clip)``
    """

    def __init__(self, eps: float = 1e-6, clip: float = 10.0, denom_eps: float = 1e-6):
        self.eps = eps
        self.clip = clip
        self.denom_eps = denom_eps

    def fit(self, y):
        y = self._finite(y)
        self.min_ = float(y.min())
        self.scale_ = float(y.max() - y.min() + self.denom_eps)
        return self

    def transform(self, y):
        self._check()
        u = (np.asarray(y, dtype=float) - self.min_) / self.scale_
        u = np.clip(u, self.eps, 1 - self.eps)
        return np.clip(np.log(u / (1 - u)), -self.clip, self.clip)

    def inverse_transform(self, z):
        self._check()
        return expit(np.asarray(z, dtype=float)) * self.scale_ + self.min_


def make_target_transform(value) -> TargetTransform:
    """None / name / instance -> a fresh (unfitted) TargetTransform."""
    if value is None:
        return IdentityTarget()
    if isinstance(value, str):
        return TARGET_TRANSFORMS.get(value)()
    if isinstance(value, TargetTransform):
        return value.clone()
    raise TypeError(f"Unsupported target transform: {value!r}")
