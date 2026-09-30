"""Plots. Every function returns a Figure (no ``show``/``savefig`` inside)."""
from .data import (
    plot_correlation,
    plot_distribution,
    plot_mask_heatmap,
    plot_missing_map,
    plot_missing_share,
    plot_time_series,
    plot_wind_rose,
)
from .model import (
    plot_feature_importance,
    plot_forecast,
    plot_learning_curve,
    plot_residuals,
    plot_window_metrics,
)

__all__ = [
    "plot_time_series", "plot_mask_heatmap", "plot_missing_share", "plot_missing_map", "plot_distribution",
    "plot_correlation", "plot_wind_rose", "plot_forecast", "plot_learning_curve", "plot_window_metrics",
    "plot_feature_importance", "plot_residuals",
]
