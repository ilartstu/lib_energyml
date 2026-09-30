"""Point and probabilistic metrics.

``nmae`` / ``nrmse`` / ``ncrps`` are divided by ``norm`` — the installed
capacity if it is known, otherwise the maximum of the target (see
``Dataset.reference_power``). With ``percent=True`` they are in %.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import ndtr

from ..core.registry import Registry
from ..utils.validation import as_1d

METRICS = Registry("metric")


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = as_1d(y_true), as_1d(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shapes differ: {y_true.shape} vs {y_pred.shape}")
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[ok], y_pred[ok]


def mae(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    return float(np.mean(np.abs(t - p))) if t.size else np.nan


def mse(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    return float(np.mean((t - p) ** 2)) if t.size else np.nan


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    if t.size < 2:
        return np.nan
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot == 0:  # constant truth: same convention as scikit-learn
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def bias(y_true, y_pred) -> float:
    """Mean of (prediction - truth): positive = overestimation."""
    t, p = _pair(y_true, y_pred)
    return float(np.mean(p - t)) if t.size else np.nan


def mape(y_true, y_pred, min_abs: float = 1e-9) -> float:
    """Mean absolute percentage error, skipping points with |y_true| <= min_abs (night zeros)."""
    t, p = _pair(y_true, y_pred)
    keep = np.abs(t) > min_abs
    return float(np.mean(np.abs((t[keep] - p[keep]) / t[keep])) * 100) if keep.any() else np.nan


def smape(y_true, y_pred) -> float:
    t, p = _pair(y_true, y_pred)
    denom = np.abs(t) + np.abs(p)
    keep = denom > 0
    return float(np.mean(2 * np.abs(t[keep] - p[keep]) / denom[keep]) * 100) if keep.any() else np.nan


def nmae(y_true, y_pred, norm: float) -> float:
    return mae(y_true, y_pred) / norm


def nrmse(y_true, y_pred, norm: float) -> float:
    return rmse(y_true, y_pred) / norm


def crps_gaussian(y_true, mean, std) -> float:
    """Mean CRPS of Gaussian forecasts N(mean, std) (closed form, as properscoring)."""
    y, mu, sigma = as_1d(y_true), as_1d(mean), as_1d(std)
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sigma)
    y, mu, sigma = y[ok], mu[ok], np.maximum(sigma[ok], 1e-12)
    z = (y - mu) / sigma
    pdf = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    crps = sigma * (z * (2 * ndtr(z) - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    return float(np.mean(crps)) if crps.size else np.nan


def coverage(y_true, lower, upper) -> float:
    """Share of observations inside [lower, upper], %."""
    y, lo, hi = as_1d(y_true), as_1d(lower), as_1d(upper)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])) * 100) if ok.any() else np.nan


def interval_width(lower, upper) -> float:
    lo, hi = as_1d(lower), as_1d(upper)
    ok = np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(hi[ok] - lo[ok])) if ok.any() else np.nan


# name -> (function, what it needs besides y_true/y_pred)
for _name, _fn in {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "bias": bias,
                   "mape": mape, "smape": smape}.items():
    METRICS.register(_name, (_fn, ()))
METRICS.register("nmae", (nmae, ("norm",)))
METRICS.register("nrmse", (nrmse, ("norm",)))
METRICS.register("crps", (lambda y, p, std: crps_gaussian(y, p, std), ("std",)))
METRICS.register("ncrps", (lambda y, p, std, norm: crps_gaussian(y, p, std) / norm, ("std", "norm")))
METRICS.register("coverage", (lambda y, p, lower, upper: coverage(y, lower, upper), ("lower", "upper")))
METRICS.register("interval_width", (lambda y, p, lower, upper: interval_width(lower, upper),
                                    ("lower", "upper")))

PERCENT_METRICS = {"nmae", "nrmse", "ncrps"}


def evaluate_regression(y_true, y_pred, metrics=("mae", "rmse", "r2"), *, norm: float | None = None,
                        std=None, lower=None, upper=None, percent: bool = False) -> dict[str, float]:
    """Compute several metrics at once.

    >>> evaluate_regression(y, pred, ["nmae", "nrmse", "r2"], norm=ds.capacity, percent=True)
    >>> evaluate_regression(y, pred, ["nmae", "nrmse", "r2", "ncrps"], norm=1.0, std=std)
    """
    context = {"norm": norm, "std": std, "lower": lower, "upper": upper}
    out = {}
    for name in metrics:
        fn, needs = METRICS.get(name)
        missing = [k for k in needs if context[k] is None]
        if missing:
            raise ValueError(f"Metric {name!r} needs {missing}")
        value = fn(y_true, y_pred, **{k: context[k] for k in needs})
        if percent and name in PERCENT_METRICS:
            value *= 100
        out[name] = value
    return out


def register_metric(name: str, fn: Callable, needs: tuple[str, ...] = ()) -> None:
    """Add a custom metric ``fn(y_true, y_pred, **needs)``."""
    METRICS.register(name, (fn, tuple(needs)))
