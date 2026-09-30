"""Plots of forecasts and training (same look as the notebooks)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core.history import TrainingHistory


def _x(values, dates):
    if dates is not None:
        return dates
    if isinstance(values, pd.Series):
        return values.index
    return np.arange(len(values))


def plot_forecast(y_true=None, y_pred=None, lower=None, upper=None, *, dates=None, title: str = "Forecast",
                  xlabel: str = "Time (Hours)", ylabel: str = "Energy Output (MW)",
                  figsize=(10, 5)) -> Figure:
    """Truth (red), prediction (blue) and an optional interval (orange/green, shaded).

    >>> plot_forecast(split.test.y[-120:], pred[-120:], title="Last 5 days")
    """
    base = y_pred if y_pred is not None else y_true
    x = _x(base, dates)
    fig = Figure(figsize=figsize)
    ax = fig.subplots()
    if y_pred is not None:
        ax.plot(x, np.asarray(y_pred, dtype=float), label="Predicted", color="blue")
    has_interval = lower is not None and upper is not None
    if has_interval:
        ax.plot(x, np.asarray(lower, dtype=float), label="Lower Bound", color="orange")
        ax.plot(x, np.asarray(upper, dtype=float), label="Upper Bound", color="green")
    if y_true is not None:
        ax.plot(x, np.asarray(y_true, dtype=float), label="True Value", color="red")
    if has_interval:
        ax.fill_between(x, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float), alpha=0.5, color="C0")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_learning_curve(history: TrainingHistory, *, title: str | None = None,
                        ylabel: str | None = None) -> Figure:
    """Train and validation loss per epoch/round with the best step marked."""
    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()
    steps = range(1, len(history.train) + 1)
    ax.plot(steps, history.train, label="Train", color="tab:blue")
    if history.has_val:
        ax.plot(range(1, len(history.val) + 1), history.val, label="Val", color="tab:orange", linestyle="--")
    if history.best_step is not None:
        ax.axvline(history.best_step, color="gray", linestyle=":",
                   label=f"Best {history.step_name} = {history.best_step}")
    ax.set_xlabel(history.step_name.capitalize())
    ax.set_ylabel(ylabel or history.metric)
    ax.set_title(title or "Learning curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_window_metrics(window_metrics: pd.DataFrame, metric: str = "nrmse", *, title: str | None = None,
                        label: str | None = None) -> Figure:
    """A metric per window over time (``evaluation.evaluate_by_window`` output)."""
    fig = Figure(figsize=(10, 5))
    ax = fig.subplots()
    ax.plot(window_metrics["start"], window_metrics[metric], marker="o", color="tab:blue",
            label=label or metric)
    ax.set_xlabel("Window start")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} per window")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_feature_importance(importance: pd.Series, *, top: int = 20, title: str = "Feature importance") -> Figure:
    values = importance.sort_values(ascending=False).head(top)[::-1]
    fig = Figure(figsize=(8, max(3, 0.35 * len(values) + 1)))
    ax = fig.subplots()
    ax.barh(values.index.astype(str), values.to_numpy(), color="tab:blue")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_residuals(y_true, y_pred, *, bins: int = 50, title: str = "Residuals") -> Figure:
    """Residuals (prediction - truth) over time and their histogram."""
    y_true = pd.Series(y_true) if not isinstance(y_true, pd.Series) else y_true
    residual = pd.Series(np.asarray(y_pred, dtype=float) - y_true.to_numpy(dtype=float), index=y_true.index)
    fig = Figure(figsize=(14, 4.5))
    ax_time, ax_hist = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 1]})
    ax_time.plot(residual.index, residual.to_numpy(), color="tab:blue", linewidth=0.8)
    ax_time.axhline(0, color="gray", linewidth=1)
    ax_time.set_title(title)
    ax_time.grid(alpha=0.3)
    ax_hist.hist(residual.dropna().to_numpy(), bins=bins, color="tab:blue", orientation="horizontal")
    ax_hist.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
