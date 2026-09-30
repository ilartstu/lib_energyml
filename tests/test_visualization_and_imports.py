import subprocess
import sys

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from energyprime.analysis import correlation, missing_map, missing_mask
from energyprime.core import TrainingHistory
from energyprime.evaluation import evaluate_by_window
from energyprime.visualization import (
    plot_correlation,
    plot_distribution,
    plot_feature_importance,
    plot_forecast,
    plot_learning_curve,
    plot_mask_heatmap,
    plot_missing_map,
    plot_missing_share,
    plot_residuals,
    plot_time_series,
    plot_wind_rose,
    plot_window_metrics,
)


def test_all_plots_return_figures(solar_frame):
    frame = solar_frame.copy()
    frame.iloc[10:20, 0] = np.nan
    mask = missing_mask(frame, nan=True, zeros=True)
    y, p = frame["POWER"].iloc[-120:], frame["POWER"].iloc[-120:] * 0.9
    figures = [
        plot_time_series(frame, highlight={"outliers": mask}),
        plot_mask_heatmap(mask),
        plot_missing_share(frame),
        plot_missing_map(missing_map(frame, buckets=20)),
        plot_distribution(frame, kind="hist"),
        plot_distribution(frame, kind="box"),
        plot_correlation(correlation(frame)),
        plot_wind_rose(frame["T_AIR"].abs(), frame["DIRECTION"]),
        plot_forecast(y, p, p * 0.9, p * 1.1),
        plot_learning_curve(TrainingHistory(train=[3, 2, 1], val=[3, 2.5, 2.6], best_step=2)),
        plot_window_metrics(evaluate_by_window(y, p, window=24, norm=50)),
        plot_feature_importance(pd.Series({"a": 1.0, "b": 2.0})),
        plot_residuals(y, p),
    ]
    assert all(isinstance(f, Figure) for f in figures)


def test_import_does_not_load_heavy_frameworks():
    code = ("import sys, energyprime, energyprime.models, energyprime.evaluation, energyprime.visualization;"
            "heavy = [m for m in ('torch', 'xgboost', 'keras', 'tensorflow', 'gpytorch', 'sklearn') "
            "if m in sys.modules]; print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
