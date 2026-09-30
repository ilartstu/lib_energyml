"""Composition of transformers."""
from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from .base import BaseTransformer, NotFittedError


class Pipeline(BaseTransformer):
    """Run transformers one after another; a Pipeline is itself a transformer.

    >>> Pipeline([TimeProcessor(), MissingValueProcessor(...), CyclicEncoder({"hour": 24})])
    >>> Pipeline([("time", TimeProcessor()), ("gaps", DeleteLargeGaps(25))])
    """

    def __init__(self, steps: Iterable[Any]):
        self.steps = list(steps)

    @property
    def named_steps(self) -> list[tuple[str, BaseTransformer]]:
        named, seen = [], {}
        for step in self.steps:
            if isinstance(step, tuple):
                name, transformer = step
            else:
                name, transformer = type(step).__name__.lower(), step
            count = seen.get(name, 0)
            seen[name] = count + 1
            named.append((f"{name}_{count}" if count else name, transformer))
        return named

    @property
    def stateless(self) -> bool:  # type: ignore[override]
        return all(t.stateless for _, t in self.named_steps)

    @property
    def changes_rows(self) -> bool:  # type: ignore[override]
        return any(t.changes_rows for _, t in self.named_steps)

    def _fit(self, df: pd.DataFrame) -> None:
        for _, transformer in self.named_steps:
            df = transformer.fit(df).transform(df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        report = {}
        for name, transformer in self.named_steps:
            df = transformer.transform(df)
            report[name] = transformer.get_report()
        self.report_ = report
        return df

    def __getitem__(self, key: int | str) -> BaseTransformer:
        named = self.named_steps
        if isinstance(key, int):
            return named[key][1]
        for name, transformer in named:
            if name == key:
                return transformer
        raise KeyError(key)

    def __len__(self) -> int:
        return len(self.steps)


class PerSeries(BaseTransformer):
    """Apply a transformer separately to every series (station, turbine...).

    Gap search, interpolation and lags then never mix neighbouring turbines.

    >>> PerSeries(LinearInterpolation(max_gap=24), by="turbine_id")
    """

    def __init__(self, transformer: BaseTransformer, by: str):
        self.transformer = transformer
        self.by = by

    @property
    def stateless(self) -> bool:  # type: ignore[override]
        return self.transformer.stateless

    @property
    def changes_rows(self) -> bool:  # type: ignore[override]
        return self.transformer.changes_rows

    def _fit(self, df: pd.DataFrame) -> None:
        self.transformers_ = {}
        for key, group in df.groupby(self.by, sort=False):
            self.transformers_[key] = self.transformer.clone().fit(group)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        fitted = getattr(self, "transformers_", {})
        parts, report = [], {}
        for key, group in df.groupby(self.by, sort=False):
            transformer = fitted.get(key)
            if transformer is None:
                if not self.transformer.stateless:
                    raise NotFittedError(f"No fitted transformer for series {key!r}")
                transformer = self.transformer.clone()
            parts.append(transformer.transform(group))
            report[key] = transformer.get_report()
        self.report_ = report
        if not parts:
            return df
        return pd.concat(parts).sort_index(kind="stable")
