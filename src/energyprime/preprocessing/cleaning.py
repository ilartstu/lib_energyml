"""Cleaner: the whole cleaning of historical data from one config.

The ``cleaning`` section of a station spec::

    cleaning:
      time: {duplicates: mean}                  # TimeProcessor options
      exclude:                                  # manual exclusions -> NaN
        - {start: "2020-06-01", end: "2020-06-03", columns: [POWER]}
      missing:                                  # run-length rules (see missing.py)
        defaults:
          zeros: true
          runs: {keep: [1, 3], interpolate: [4, 24], drop_rows: [25, null]}
        columns:
          T_AIR: {zeros: false}
      outliers:
        detectors: {zscore: {threshold: 5.0}, iqr: {factor: 4.0}, hampel: {window: 72, n_sigmas: 12}}
        apply: [zscore]                         # physical limits are always applied
        action: interpolate
      interpolation: linear                     # or "time"

Order of the steps (always the same):

1. time axis (sort, duplicate timestamps);
2. manual exclusions -> NaN;
3. ``nan_to_zero`` where configured;
4. rows inside ``drop_rows`` runs are removed (only target/feature columns can
   remove rows);
5. outliers are detected (statistics come from ``fit``, i.e. training data);
6. selected runs and outliers are filled (interpolation etc.);
7. report: what was removed, filled and what is still missing.
"""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.quality import removed_periods
from ..core.base import BaseTransformer
from ..core.schema import StationSpec
from ..utils.validation import as_frame, numeric_columns
from .missing import build_rules, drop_masks, fill_frame, plan_table
from .outliers import OutlierProcessor
from .time import TimeProcessor

CONFIG_KEYS = {"time", "exclude", "missing", "outliers", "interpolation"}


class Cleaner(BaseTransformer):
    """Clean historical station data according to a config (see module docstring).

    ``fit`` learns outlier statistics (z-score mean/std...) — call it on the
    training part to avoid leaking information from the test period.

    >>> cleaner = Cleaner.from_spec(ds.spec)
    >>> split = cleaner.fit(split.train).transform(split)   # statistics from train only
    >>> cleaner.get_report()["fills"]
    """

    stateless = False
    changes_rows = True

    def __init__(self, config: dict | None = None, limits: dict[str, Any] | None = None,
                 columns: list[str] | None = None, drop_scope: list[str] | None = None):
        self.config = config
        self.limits = limits
        self.columns = columns
        self.drop_scope = drop_scope

    @classmethod
    def from_spec(cls, spec: StationSpec | dict | str, **overrides: Any) -> "Cleaner":
        """Cleaner from a station spec: its ``cleaning`` section and column limits."""
        spec = StationSpec.load(spec)
        params = {
            "config": copy.deepcopy(spec.cleaning),
            "limits": spec.limits,
            "drop_scope": spec.used_columns or None,
        }
        params.update(overrides)
        return cls(**params)

    # ---------------------------------------------------------------- config
    def _config(self) -> dict:
        config = self.config or {}
        unknown = set(config) - CONFIG_KEYS
        if unknown:
            raise ValueError(f"Unknown cleaning option(s) {sorted(unknown)}; allowed: {sorted(CONFIG_KEYS)}")
        return config

    def _rules(self, df: pd.DataFrame):
        cols = self.columns if self.columns is not None else numeric_columns(df)
        return build_rules(self._config().get("missing"), [c for c in cols if c in df.columns])

    def _method(self) -> str:
        return self._config().get("interpolation", "linear")

    def _prepare(self, df: pd.DataFrame, rules) -> tuple[pd.DataFrame, dict, int]:
        """Steps 1-3: time axis, exclusions, nan_to_zero."""
        config = self._config()
        time_step = TimeProcessor(**(config.get("time") or {}))
        df = time_step.transform(df)
        excluded = 0
        for item in config.get("exclude") or []:
            unknown = set(item) - {"start", "end", "columns"}
            if unknown:
                raise ValueError(f"Unknown key(s) {sorted(unknown)} in an 'exclude' item")
            rows = (df.index >= pd.Timestamp(item.get("start", df.index.min()))) & \
                   (df.index <= pd.Timestamp(item.get("end", df.index.max())))
            cols = item.get("columns") or list(rules)
            excluded += int(rows.sum()) * len(cols)
            df.loc[rows, cols] = np.nan
        for col, col_rules in rules.items():
            if col_rules.nan_to_zero:
                df[col] = df[col].fillna(0.0)
        return df, time_step.get_report(), excluded

    def _outlier_processor(self, rules) -> OutlierProcessor:
        config = dict(self._config().get("outliers") or {})
        unknown = set(config) - {"detectors", "apply", "action", "columns"}
        if unknown:
            raise ValueError(f"Unknown key(s) {sorted(unknown)} in 'outliers'")
        detectors = config.get("detectors")
        if detectors is None:
            detectors = {}
        return OutlierProcessor(
            detectors=detectors,
            apply=config.get("apply"),
            limits=self.limits,
            columns=config.get("columns") or list(rules),
            action=config.get("action", "interpolate"),
            ignore_zeros=[c for c, r in rules.items() if r.zeros],
            method=self._method(),
        )

    # -------------------------------------------------------------- fitting
    def _fit(self, df: pd.DataFrame) -> None:
        rules = self._rules(df)
        df, _, _ = self._prepare(df, rules)
        drop, _, _ = drop_masks(df, rules, self.drop_scope)
        df = df.loc[~drop.to_numpy()]
        self.outliers_ = self._outlier_processor(rules).fit(df)

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rules = self._rules(df)
        rows_before = len(df)
        df, time_report, excluded = self._prepare(df, rules)

        drop, per_column, blocks = drop_masks(df, rules, self.drop_scope)
        periods = removed_periods(drop, per_column)
        df = df.loc[~drop.to_numpy()]

        masks = self.outliers_.detect(df)
        outlier_mask = masks["applied"]
        extra = None
        if self.outliers_.action == "interpolate":
            extra = outlier_mask
        else:
            df = self.outliers_.transform(df)

        df, fills = fill_frame(df, rules, extra_interpolate=extra, method=self._method())
        used = [c for c in rules if c in df.columns]
        self.report_ = {
            "time": time_report,
            "excluded_values": excluded,
            "rows_before": rows_before,
            "rows_dropped": int(drop.sum()),
            "rows_after": len(df),
            "blocks": blocks,
            "removed_periods": periods,
            "outliers": OutlierProcessor.diagnose_from(masks),
            "outlier_mask": outlier_mask,
            "fills": fills,
            "remaining_nan": df[used].isna().sum(),
        }
        return df

    def plan(self, data) -> dict:
        """Dry run of the missing-value rules: runs per rule and rows that would be removed.

        Does not change the data and does not need ``fit``.
        """
        df = as_frame(data).copy()
        rules = self._rules(df)
        df, _, _ = self._prepare(df, rules)
        drop, per_column, _ = drop_masks(df, rules, self.drop_scope)
        return {
            "runs": plan_table(df, rules),
            "rows_to_drop": int(drop.sum()),
            "removed_periods": removed_periods(drop, per_column),
        }
