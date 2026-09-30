"""Contiguous runs of ``True`` in a boolean mask (runs of NaN, zeros, gaps...).

Everything in the library that talks about "N missing values in a row" is built
on these helpers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def runs_of(mask) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` (inclusive positions) for each run of True."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    starts = edges[0::2]
    ends = edges[1::2] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def run_lengths(mask) -> np.ndarray:
    """For every position: the length of the True-run it belongs to (0 for False)."""
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros(mask.size, dtype=np.int64)
    for start, end in runs_of(mask):
        out[start:end + 1] = end - start + 1
    return out


def runs_table(mask, index=None) -> pd.DataFrame:
    """Runs of True as a table: positions, length and (optionally) index labels."""
    rows = []
    for start, end in runs_of(mask):
        row = {"start_pos": start, "end_pos": end, "length": end - start + 1}
        if index is not None:
            row["start"] = index[start]
            row["end"] = index[end]
        rows.append(row)
    columns = ["start_pos", "end_pos", "length"] + (["start", "end"] if index is not None else [])
    return pd.DataFrame(rows, columns=columns)
