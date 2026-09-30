import numpy as np
import pandas as pd

from energyprime.analysis import (
    compare_missing,
    describe_columns,
    find_blocks,
    missing_map,
    run_length_table,
)
from energyprime.utils import run_lengths, runs_of, runs_table


def test_runs_of_and_lengths():
    mask = [False, True, True, False, True, False, False, True, True, True]
    assert runs_of(mask) == [(1, 2), (4, 4), (7, 9)]
    assert run_lengths(mask).tolist() == [0, 2, 2, 0, 1, 0, 0, 3, 3, 3]
    assert runs_of([]) == []
    table = runs_table(mask)
    assert table["length"].tolist() == [2, 1, 3]


def _frame():
    index = pd.date_range("2021-01-01", periods=12, freq="h")
    return pd.DataFrame({
        "A": [1, np.nan, np.nan, 0, 5, 0, 0, 0, 7, np.nan, 2, 3],
        "B": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }, index=index)


def test_describe_columns_counts():
    stats = describe_columns(_frame())
    assert stats.loc["A", "nan"] == 3
    assert stats.loc["A", "zeros"] == 4
    assert stats.loc["B", "nan"] == 0
    assert stats.loc["B", "max"] == 12


def test_run_length_table_merged_and_separate():
    merged = run_length_table(_frame(), ["A"], nan=True, zeros=True, merge=True)
    # runs: [NaN NaN 0] = 3, [0 0 0] = 3, [NaN] = 1
    assert dict(zip(merged["length"], merged["runs"])) == {1: 1, 3: 2}
    separate = run_length_table(_frame(), ["A"], merge=False)
    nan_runs = separate[separate["kind"] == "nan"]
    assert dict(zip(nan_runs["length"], nan_runs["runs"])) == {1: 1, 2: 1}


def test_find_blocks_kinds():
    blocks = find_blocks(_frame(), min_length=3, columns=["A"])
    assert blocks["kind"].tolist() == ["nan+zero", "zero"]
    assert blocks["length"].tolist() == [3, 3]
    assert blocks.loc[0, "n_nan"] == 2 and blocks.loc[0, "n_zeros"] == 1


def test_missing_map_and_compare():
    df = _frame()
    shares = missing_map(df, buckets=3, nan=True, zeros=True)
    assert shares.shape == (3, 2)
    assert np.isclose(shares["A"].iloc[0], 3 / 4)
    after = df.fillna(1.0)
    comparison = compare_missing(df, after, nan=True, zeros=False)
    assert comparison.loc["A", "before"] == 3 and comparison.loc["A", "after"] == 0
