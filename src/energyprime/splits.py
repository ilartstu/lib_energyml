"""Splitting by time: train / validation / test, and expanding-window folds."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .utils.validation import as_frame, check_time_index


@dataclass
class Split:
    """Three consecutive parts of the same data (Dataset or DataFrame).

    Transformers accept a Split directly: ``cleaner.fit(split)`` fits on
    ``train``; ``cleaner.transform(split)`` returns a new Split.
    """

    train: Any
    val: Any
    test: Any

    __energyprime_split__ = True

    def map(self, fn: Callable[[Any], Any]) -> "Split":
        return Split(fn(self.train), None if self.val is None else fn(self.val), fn(self.test))

    def replace(self, **parts: Any) -> "Split":
        """New Split with some parts replaced: ``split.replace(test=new_test)``."""
        return dataclasses.replace(self, **parts)

    def parts(self) -> dict[str, Any]:
        return {name: part for name, part in (("train", self.train), ("val", self.val),
                                              ("test", self.test)) if part is not None}

    def boundaries(self) -> pd.DataFrame:
        rows = []
        for name, part in self.parts().items():
            frame = as_frame(part)
            rows.append({"part": name, "start": frame.index.min(), "end": frame.index.max(),
                         "rows": len(frame)})
        return pd.DataFrame(rows).set_index("part")

    def __repr__(self) -> str:
        sizes = ", ".join(f"{k}={len(as_frame(v))}" for k, v in self.parts().items())
        return f"Split({sizes})"


def _slice(data: Any, frame: pd.DataFrame) -> Any:
    if hasattr(data, "with_data") and not isinstance(data, pd.DataFrame):
        return data.with_data(frame)
    return frame


def _start_of_tail(frame: pd.DataFrame, size, end_pos: int) -> int:
    """Position where a tail of ``size`` ends at ``end_pos`` (exclusive) begins."""
    if isinstance(size, int):
        return max(end_pos - size, 0)
    if isinstance(size, (pd.Timestamp,)) or (isinstance(size, str) and _looks_like_date(size)):
        return int(frame.index[:end_pos].searchsorted(pd.Timestamp(size), side="left"))
    last = frame.index[end_pos - 1]
    cutoff = last - pd.Timedelta(size)
    return int(frame.index[:end_pos].searchsorted(cutoff, side="right"))


def _looks_like_date(text: str) -> bool:
    try:
        pd.Timedelta(text)
        return False
    except ValueError:
        return True


def split_by_time(data: Any, test, val=None) -> Split:
    """Split the end of a series into test (last) and validation (before test).

    ``test`` / ``val`` can be:

    * a duration — ``"30D"`` means the last 30 days;
    * a number of rows — ``720`` (as in the notebooks: ``eval_days * 24``);
    * a timestamp — ``"2021-12-01"``: the part starts there.

    >>> split = split_by_time(ds, test="30D", val="5D")
    >>> split = split_by_time(ds, test=720, val=120)       # notebook-compatible
    """
    frame = as_frame(data)
    check_time_index(frame)
    n = len(frame)
    test_start = _start_of_tail(frame, test, n)
    val_start = test_start if val is None else _start_of_tail(frame, val, test_start)
    if val_start <= 0:
        raise ValueError("Not enough data for the requested test/validation sizes")
    train = _slice(data, frame.iloc[:val_start])
    val_part = None if val is None else _slice(data, frame.iloc[val_start:test_start])
    return Split(train, val_part, _slice(data, frame.iloc[test_start:]))


def expanding_window_splits(data: Any, n_splits: int, test, val=None, step=None) -> list[Split]:
    """Rolling-origin folds: the training part grows, the test window moves forward.

    Each fold's test has size ``test``; consecutive folds are ``step`` apart
    (default: ``test``). The last fold ends at the end of the data.
    """
    frame = as_frame(data)
    step = test if step is None else step
    splits = []
    end = len(frame)
    for _ in range(n_splits):
        part = frame.iloc[:end]
        split = split_by_time(_slice(data, part), test=test, val=val)
        splits.append(split)
        new_end = _start_of_tail(part, step, end)
        if new_end <= 0 or new_end == end:
            break
        end = new_end
    return list(reversed(splits))
