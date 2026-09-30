"""Data quality: how many NaN and zeros, where they are, how long the runs are.

These functions only look at the data; they never change it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.runs import runs_of
from ..utils.validation import as_frame


def _columns(df: pd.DataFrame, columns) -> list[str]:
    if columns is None:
        return list(df.columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return list(columns)


def _numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def nan_mask(series: pd.Series) -> np.ndarray:
    return ~np.isfinite(_numeric(series))


def zero_mask(series: pd.Series) -> np.ndarray:
    """Exactly zero (0.001 is not zero)."""
    values = _numeric(series)
    return np.isfinite(values) & (values == 0)


def missing_mask(data, columns=None, *, nan: bool = True, zeros: bool = False) -> pd.DataFrame:
    """Boolean frame: True where a value is NaN (and/or exactly zero)."""
    df = as_frame(data)
    out = {}
    for col in _columns(df, columns):
        mask = np.zeros(len(df), dtype=bool)
        if nan:
            mask |= nan_mask(df[col])
        if zeros:
            mask |= zero_mask(df[col])
        out[col] = mask
    return pd.DataFrame(out, index=df.index)


def column_stats(series: pd.Series) -> dict:
    """Counts of NaN / zeros and basic statistics of one column."""
    n = int(len(series))
    numeric = pd.to_numeric(series, errors="coerce")
    present = int(series.notna().sum())
    is_numeric = (pd.api.types.is_numeric_dtype(series) if present == 0
                  else numeric.notna().sum() / present >= 0.8)
    nan_count = int(series.isna().sum()) if not is_numeric else int(numeric.isna().sum())
    stats = {
        "n": n,
        "nan": nan_count,
        "nan_pct": round(100 * nan_count / n, 4) if n else 0.0,
        "zeros": 0,
        "zeros_pct": 0.0,
        "min": np.nan, "max": np.nan, "mean": np.nan, "median": np.nan, "std": np.nan,
        "is_numeric": bool(is_numeric),
    }
    if is_numeric:
        zeros = int((numeric == 0).sum())
        finite = numeric[np.isfinite(numeric)]
        stats.update({
            "zeros": zeros,
            "zeros_pct": round(100 * zeros / n, 4) if n else 0.0,
            "min": finite.min(), "max": finite.max(), "mean": finite.mean(),
            "median": finite.median(), "std": finite.std(),
        })
    return stats


def describe_columns(data, columns=None) -> pd.DataFrame:
    """One row per column: points, NaN, NaN %, zeros, zeros %, min, max, mean, median, std."""
    df = as_frame(data)
    rows = {col: column_stats(df[col]) for col in _columns(df, columns)}
    return pd.DataFrame.from_dict(rows, orient="index")


def _kind_masks(series: pd.Series, nan: bool, zeros: bool, merge: bool) -> list[tuple[str, np.ndarray]]:
    n_mask = nan_mask(series) if nan else np.zeros(len(series), dtype=bool)
    z_mask = zero_mask(series) if zeros else np.zeros(len(series), dtype=bool)
    if merge:
        kind = "nan_or_zero" if (nan and zeros) else ("nan" if nan else "zero")
        return [(kind, n_mask | z_mask)]
    return [(k, m) for k, m, on in (("nan", n_mask, nan), ("zero", z_mask, zeros)) if on]


def run_length_table(data, columns=None, *, nan: bool = True, zeros: bool = True,
                     merge: bool = True) -> pd.DataFrame:
    """How many runs of each length every column has.

    Use it to choose the thresholds of the cleaning rules: e.g. "POWER has 120
    single NaN, 30 runs of 2, 4 runs longer than 24".

    ``merge=True``: NaN and zeros next to each other form one run.
    """
    df = as_frame(data)
    rows = []
    for col in _columns(df, columns):
        for kind, mask in _kind_masks(df[col], nan, zeros, merge):
            lengths = pd.Series([e - s + 1 for s, e in runs_of(mask)], dtype=int)
            for length, count in lengths.value_counts().sort_index().items():
                rows.append({"column": col, "kind": kind, "length": int(length),
                             "runs": int(count), "points": int(length * count)})
    return pd.DataFrame(rows, columns=["column", "kind", "length", "runs", "points"])


def find_blocks(data, min_length: int = 1, columns=None, *, nan: bool = True, zeros: bool = True,
                merge: bool = True) -> pd.DataFrame:
    """Runs of NaN/zeros not shorter than ``min_length``: where they start and end."""
    df = as_frame(data)
    rows = []
    for col in _columns(df, columns):
        n_mask = nan_mask(df[col])
        z_mask = zero_mask(df[col])
        for kind, mask in _kind_masks(df[col], nan, zeros, merge):
            for start, end in runs_of(mask):
                length = end - start + 1
                if length < min_length:
                    continue
                n_nan = int(n_mask[start:end + 1].sum())
                n_zero = int(z_mask[start:end + 1].sum())
                if kind == "nan_or_zero":
                    kind_label = "nan+zero" if (n_nan and n_zero) else ("nan" if n_nan else "zero")
                else:
                    kind_label = kind
                rows.append({"column": col, "kind": kind_label, "start": df.index[start],
                             "end": df.index[end], "length": length,
                             "n_nan": n_nan, "n_zeros": n_zero})
    frame = pd.DataFrame(rows, columns=["column", "kind", "start", "end", "length", "n_nan", "n_zeros"])
    return frame.sort_values(["start", "column"], kind="stable").reset_index(drop=True)


def removed_periods(drop_mask: pd.Series, reasons: pd.DataFrame | None = None) -> pd.DataFrame:
    """Contiguous periods of dropped rows and the columns that caused them."""
    rows = []
    index = drop_mask.index
    values = drop_mask.to_numpy(dtype=bool)
    for start, end in runs_of(values):
        cause = ""
        if reasons is not None:
            block = reasons.iloc[start:end + 1]
            cause = ", ".join(c for c in reasons.columns if block[c].any())
        rows.append({"start": index[start], "end": index[end], "rows": end - start + 1, "reason": cause})
    return pd.DataFrame(rows, columns=["start", "end", "rows", "reason"])


def missing_map(data, columns=None, buckets: int = 240, *, nan: bool = True,
                zeros: bool = False) -> pd.DataFrame:
    """Share of NaN/zeros per column in ``buckets`` equal time slices (for heatmaps)."""
    df = as_frame(data)
    mask = missing_mask(df, columns, nan=nan, zeros=zeros)
    n = len(mask)
    if n == 0:
        return mask.astype(float)
    buckets = max(1, min(int(buckets), n))
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    values = mask.to_numpy(dtype=float)
    shares = [values[a:b].mean(axis=0) if b > a else np.zeros(values.shape[1])
              for a, b in zip(edges[:-1], edges[1:])]
    return pd.DataFrame(shares, index=mask.index[edges[:-1]], columns=mask.columns)


def compare_missing(before, after, columns=None, *, nan: bool = True, zeros: bool = True) -> pd.DataFrame:
    """Share of NaN/zeros per column before and after processing."""
    b = missing_mask(before, columns, nan=nan, zeros=zeros)
    cols = [c for c in b.columns if c in as_frame(after).columns]
    a = missing_mask(after, cols, nan=nan, zeros=zeros)
    out = pd.DataFrame({
        "before": b[cols].sum(),
        "before_pct": b[cols].mean() * 100,
        "after": a.sum(),
        "after_pct": a.mean() * 100,
    })
    out["diff_pct"] = out["before_pct"] - out["after_pct"]
    return out
