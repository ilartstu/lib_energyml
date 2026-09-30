"""Exploratory statistics: correlations, autocorrelation, spectrum."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.validation import as_frame, infer_freq, numeric_columns

METHODS = ("pearson", "spearman", "kendall")


def correlation(data, columns=None, method: str = "pearson") -> pd.DataFrame:
    """Correlation matrix of numeric columns."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    df = as_frame(data)
    return df[numeric_columns(df, columns)].corr(method=method)


def pair_counts(data, columns=None) -> pd.DataFrame:
    """Number of rows where both columns are present (the size behind each correlation)."""
    df = as_frame(data)
    valid = df[numeric_columns(df, columns)].notna().astype(int)
    return valid.T.dot(valid)


def target_correlation(data, target: str, method: str = "pearson") -> pd.Series:
    """Correlation of every numeric column with the target, strongest first."""
    corr = correlation(data, method=method)[target].drop(target)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def acf(series: pd.Series, max_lag: int = 168) -> pd.Series:
    """Autocorrelation for lags 0..max_lag (in steps). Peaks hint at useful lags / window sizes."""
    values = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both").to_numpy(float)
    values = np.nan_to_num(values - np.nanmean(values))
    n = values.size
    max_lag = int(min(max_lag, n - 1))
    denom = float(np.dot(values, values)) or 1.0
    result = [float(np.dot(values[:n - k], values[k:]) / denom) for k in range(max_lag + 1)]
    return pd.Series(result, index=pd.RangeIndex(max_lag + 1, name="lag"), name=series.name)


def spectrum(series: pd.Series) -> pd.DataFrame:
    """FFT periodogram: power per frequency, with the period in steps and hours."""
    values = pd.to_numeric(series, errors="coerce").interpolate(limit_direction="both").to_numpy(float)
    values = np.nan_to_num(values - np.nanmean(values))
    power = np.abs(np.fft.rfft(values)) ** 2
    freq = np.fft.rfftfreq(values.size)
    power, freq = power[1:], freq[1:]
    period_steps = 1.0 / freq
    step = infer_freq(series.index) if isinstance(series.index, pd.DatetimeIndex) else None
    period_hours = period_steps * step.total_seconds() / 3600 if step is not None else np.full_like(freq, np.nan)
    return pd.DataFrame({"frequency": freq, "period_steps": period_steps,
                         "period_hours": period_hours, "power": power})
