"""Plots of the raw / cleaned data.

Every function returns a :class:`matplotlib.figure.Figure` and never calls
``show()`` or ``savefig()``: in Jupyter the returned figure is displayed,
elsewhere save it with ``fig.savefig(path, dpi=300)``.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from ..analysis.quality import missing_mask
from ..utils.validation import as_frame, numeric_columns

LINE_COLOR = "#2a9d8f"
HIGHLIGHT_COLORS = ["black", "tab:orange", "tab:purple", "tab:blue", "tab:brown"]


def _labels(columns, labels: dict | None) -> list[str]:
    labels = labels or {}
    return [labels.get(c, c) for c in columns]


def _date_axis(ax) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_time_series(data, columns=None, *, missing: str | None = "nan_or_zero",
                     highlight: dict[str, pd.DataFrame | pd.Series] | None = None, title: str | None = None,
                     labels: dict[str, str] | None = None, missing_label: str = "0 / NaN",
                     height_per_row: float = 2.2, width: float = 18) -> Figure:
    """One panel per column; red dots at NaN/zeros, extra masks (e.g. outliers) highlighted.

    >>> plot_time_series(ds, missing="nan_or_zero", highlight={"outliers": report["outlier_mask"]})
    """
    df = as_frame(data)
    cols = list(columns) if columns is not None else numeric_columns(df)
    nan_on = missing in ("nan", "nan_or_zero")
    zero_on = missing in ("zero", "nan_or_zero")
    fig = Figure(figsize=(width, max(6, height_per_row * len(cols))))
    axes = fig.subplots(len(cols), 1, sharex=True, squeeze=False)[:, 0]
    for ax, col, label in zip(axes, cols, _labels(cols, labels)):
        y = pd.to_numeric(df[col], errors="coerce")
        ax.plot(df.index, y, color=LINE_COLOR, linewidth=0.6, alpha=0.75)
        valid = y.dropna()
        if len(valid):
            ymin, ymax = valid.min(), valid.max()
            padding = 1 if ymin == ymax else (ymax - ymin) * 0.08
            ax.set_ylim(ymin - padding, ymax + padding)
            if missing:
                mask = missing_mask(df[[col]], nan=nan_on, zeros=zero_on)[col].to_numpy()
                if mask.any():
                    ax.scatter(df.index[mask], y[mask].fillna(ymin), color="red", s=12, zorder=3,
                               label=missing_label)
            for (name, extra), color in zip((highlight or {}).items(), HIGHLIGHT_COLORS):
                extra_mask = extra[col] if isinstance(extra, pd.DataFrame) and col in extra else \
                    (extra if isinstance(extra, pd.Series) else None)
                if extra_mask is None:
                    continue
                extra_mask = extra_mask.reindex(df.index, fill_value=False).to_numpy(dtype=bool)
                if extra_mask.any():
                    ax.scatter(df.index[extra_mask], y[extra_mask], color=color, s=20, zorder=4, label=name)
        ax.set_ylabel(label, rotation=0, labelpad=50, ha="right", va="center")
        ax.grid(True, alpha=0.3)
    if title:
        axes[0].set_title(title, fontsize=14)
    handles, names = [], []
    for ax in axes:
        for h, n in zip(*ax.get_legend_handles_labels()):
            if n not in names:
                handles.append(h)
                names.append(n)
    if handles:
        axes[0].legend(handles, names, loc="upper right")
    _date_axis(axes[-1])
    fig.tight_layout()
    return fig


def plot_mask_heatmap(mask: pd.DataFrame, columns=None, *, title: str | None = None,
                      labels: dict[str, str] | None = None, color: str = "red",
                      ylabel: str = "Parameters") -> Figure:
    """White/colored map of a boolean frame (NaN/zeros, selected outliers...) over time."""
    cols = list(columns) if columns is not None else list(mask.columns)
    data = mask[cols].T.astype(int).to_numpy()
    fig = Figure(figsize=(18, max(6, 0.35 * len(cols))))
    ax = fig.subplots()
    ax.imshow(data, aspect="auto", interpolation="nearest", cmap=ListedColormap(["white", color]),
              vmin=0, vmax=1)
    if title:
        ax.set_title(title, fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(_labels(cols, labels))
    if len(mask):
        ticks = np.linspace(0, len(mask) - 1, min(14, len(mask)), dtype=int)
        ax.set_xticks(ticks)
        index = mask.index
        texts = index[ticks].strftime("%b %Y") if isinstance(index, pd.DatetimeIndex) else index[ticks]
        ax.set_xticklabels(texts, rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_missing_share(data, columns=None, *, nan: bool = True, zeros: bool = True, threshold: float = 10.0,
                       title: str = "Share of zeros and NaN", labels: dict[str, str] | None = None) -> Figure:
    """Horizontal bars: % of NaN/zeros per column, with a threshold line."""
    df = as_frame(data)
    cols = list(columns) if columns is not None else numeric_columns(df)
    share = missing_mask(df, cols, nan=nan, zeros=zeros).mean() * 100
    fig = Figure(figsize=(12, max(5, 0.7 * len(cols))))
    ax = fig.subplots()
    bars = ax.barh(_labels(cols, labels), share[cols].to_numpy(), color=LINE_COLOR)
    ax.axvline(threshold, color="gray", linestyle="--", linewidth=1, label=f"threshold {threshold:g}%")
    top = max(threshold, float(share.max()) if len(share) else 0.0)
    ax.set_xlim(0, top * 1.15)
    for bar, value in zip(bars, share[cols].to_numpy()):
        ax.text(bar.get_width() + top * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%",
                va="center", fontsize=10)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("%")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_missing_map(shares: pd.DataFrame, *, title: str = "Share of missing values over time",
                     labels: dict[str, str] | None = None) -> Figure:
    """Heatmap of ``analysis.missing_map`` output (0..1 per time bucket)."""
    fig = Figure(figsize=(18, max(4, 0.35 * shares.shape[1])))
    ax = fig.subplots()
    image = ax.imshow(shares.T.to_numpy(), aspect="auto", interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
    ax.set_yticks(np.arange(shares.shape[1]))
    ax.set_yticklabels(_labels(shares.columns, labels))
    if len(shares):
        ticks = np.linspace(0, len(shares) - 1, min(14, len(shares)), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels(pd.DatetimeIndex(shares.index[ticks]).strftime("%b %Y"), rotation=45, ha="right")
    fig.colorbar(image, ax=ax, label="share")
    ax.set_title(title, fontsize=14)
    fig.tight_layout()
    return fig


def plot_distribution(data, columns=None, *, kind: str = "hist", bins: int = 50,
                      labels: dict[str, str] | None = None) -> Figure:
    """Histograms (``kind="hist"``) or box plots (``kind="box"``) of columns."""
    df = as_frame(data)
    cols = list(columns) if columns is not None else numeric_columns(df)
    if kind == "box":
        fig = Figure(figsize=(max(6, 1.2 * len(cols)), 5))
        ax = fig.subplots()
        ax.boxplot([pd.to_numeric(df[c], errors="coerce").dropna() for c in cols])
        ax.set_xticks(range(1, len(cols) + 1))
        ax.set_xticklabels(_labels(cols, labels), rotation=45, ha="right")
        ax.grid(alpha=0.3)
    elif kind == "hist":
        ncols = min(3, len(cols))
        nrows = int(np.ceil(len(cols) / ncols))
        fig = Figure(figsize=(5 * ncols, 3.5 * nrows))
        axes = fig.subplots(nrows, ncols, squeeze=False).ravel()
        for ax, col, label in zip(axes, cols, _labels(cols, labels)):
            ax.hist(pd.to_numeric(df[col], errors="coerce").dropna(), bins=bins, color=LINE_COLOR)
            ax.set_title(label)
            ax.grid(alpha=0.3)
        for ax in axes[len(cols):]:
            ax.set_visible(False)
    else:
        raise ValueError("kind must be 'hist' or 'box'")
    fig.tight_layout()
    return fig


def plot_correlation(corr: pd.DataFrame, *, annotate: bool = True, title: str = "Correlation") -> Figure:
    """Correlation matrix (``analysis.correlation``) as a heatmap."""
    n = len(corr)
    fig = Figure(figsize=(max(6, 0.7 * n + 2), max(5, 0.7 * n + 1)))
    ax = fig.subplots()
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)
    if annotate:
        for i in range(n):
            for j in range(n):
                value = corr.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_wind_rose(speed, direction, *, sectors: int = 16, speed_bins=None,
                   title: str = "Wind rose") -> Figure:
    """Share of time per direction sector, stacked by speed class."""
    speed = pd.to_numeric(pd.Series(speed), errors="coerce").to_numpy(float)
    direction = pd.to_numeric(pd.Series(direction), errors="coerce").to_numpy(float) % 360
    ok = np.isfinite(speed) & np.isfinite(direction)
    speed, direction = speed[ok], direction[ok]
    if speed_bins is None:
        speed_bins = np.quantile(speed, [0, 0.25, 0.5, 0.75, 1.0]) if speed.size else [0, 1]
    width = 360 / sectors
    sector = ((direction + width / 2) // width).astype(int) % sectors
    angles = np.deg2rad(np.arange(sectors) * width)
    fig = Figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bottom = np.zeros(sectors)
    for low, high in zip(speed_bins[:-1], speed_bins[1:]):
        in_bin = (speed >= low) & (speed <= high if high == speed_bins[-1] else speed < high)
        counts = np.bincount(sector[in_bin], minlength=sectors) / max(speed.size, 1) * 100
        ax.bar(angles, counts, width=np.deg2rad(width), bottom=bottom, label=f"{low:.1f}–{high:.1f}",
               edgecolor="white")
        bottom += counts
    ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.0), title="speed")
    ax.set_title(title)
    fig.tight_layout()
    return fig
