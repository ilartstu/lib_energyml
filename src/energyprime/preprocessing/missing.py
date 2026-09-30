"""Missing values: what counts as missing and what to do with a run of them.

Every column gets :class:`ColumnRules`::

    nan: true              # NaN is missing
    zeros: true            # an exact 0 is missing too (0.001 is not)
    merge: true            # "0 0 NaN 0" is one run of length 4
    nan_to_zero: false     # replace NaN by 0 before anything else
    runs:                  # action by run length (both bounds inclusive)
      keep: [1, 3]
      interpolate: [4, 24]
      drop_rows: [25, null]

Actions: ``keep``, ``interpolate`` (linear, only between known values),
``fill_zero``, ``ffill``, ``bfill``, ``set_nan``, ``drop_rows``. A run that
matches no rule is kept.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..analysis.quality import removed_periods
from ..core.base import BaseTransformer
from ..utils.runs import runs_of
from ..utils.validation import as_frame

ACTIONS = ("keep", "interpolate", "fill_zero", "ffill", "bfill", "set_nan", "drop_rows")
FILL_ACTIONS = ("interpolate", "fill_zero", "ffill", "bfill", "set_nan")


@dataclass
class RunRule:
    """``action`` for runs with ``min_length <= length <= max_length`` (None = no upper bound)."""

    min_length: int = 1
    max_length: int | None = None
    action: str = "keep"

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ValueError(f"Unknown action {self.action!r}; use one of {ACTIONS}")
        self.min_length = int(self.min_length)
        if self.min_length < 1:
            raise ValueError("min_length must be >= 1")
        if self.max_length is not None:
            self.max_length = int(self.max_length)
            if self.max_length < self.min_length:
                raise ValueError(f"Empty range [{self.min_length}, {self.max_length}] for {self.action!r}")

    def matches(self, length: int) -> bool:
        return length >= self.min_length and (self.max_length is None or length <= self.max_length)

    def describe(self) -> str:
        upper = "inf" if self.max_length is None else self.max_length
        return f"[{self.min_length}, {upper}]"


def parse_runs(value: Any) -> list[RunRule]:
    """Rules from ``{action: [min, max]}`` or ``[{length: [min, max], action: ...}]``."""
    if value is None:
        return []
    rules: list[RunRule] = []
    if isinstance(value, dict):
        items = value.items()
        for action, bounds in items:
            if isinstance(bounds, (list, tuple)) and bounds and isinstance(bounds[0], (list, tuple)):
                for b in bounds:  # several ranges for one action
                    rules.append(RunRule(b[0], b[1], action))
            else:
                rules.append(RunRule(bounds[0], bounds[1], action))
    else:
        for item in value:
            if isinstance(item, RunRule):
                rules.append(item)
            else:
                low, high = item["length"]
                rules.append(RunRule(low, high, item["action"]))
    _check_overlaps(rules)
    return sorted(rules, key=lambda r: r.min_length)


def _check_overlaps(rules: list[RunRule]) -> None:
    ordered = sorted(rules, key=lambda r: r.min_length)
    for a, b in zip(ordered, ordered[1:]):
        if a.max_length is None or a.max_length >= b.min_length:
            raise ValueError(
                f"Run rules overlap: {a.action} {a.describe()} and {b.action} {b.describe()}"
            )


@dataclass
class ColumnRules:
    nan: bool = True
    zeros: bool = False
    merge: bool = True
    nan_to_zero: bool = False
    runs: list[RunRule] = field(default_factory=list)
    nan_runs: list[RunRule] | None = None
    zero_runs: list[RunRule] | None = None

    _KEYS = ("nan", "zeros", "merge", "nan_to_zero", "runs", "nan_runs", "zero_runs")

    @classmethod
    def from_dict(cls, data: dict | None, base: "ColumnRules | None" = None) -> "ColumnRules":
        """Rules from a config mapping; keys not given are taken from ``base``."""
        rules = copy.deepcopy(base) if base is not None else cls()
        for key, value in (data or {}).items():
            if key not in cls._KEYS:
                raise ValueError(f"Unknown missing-value option {key!r}; allowed: {list(cls._KEYS)}")
            if key in ("runs", "nan_runs", "zero_runs"):
                value = parse_runs(value)
            setattr(rules, key, value)
        return rules

    def rules_for(self, kind: str) -> list[RunRule]:
        if kind == "nan" and self.nan_runs is not None:
            return self.nan_runs
        if kind == "zero" and self.zero_runs is not None:
            return self.zero_runs
        return self.runs

    def action_for(self, kind: str, length: int) -> tuple[str, RunRule | None]:
        for rule in self.rules_for(kind):
            if rule.matches(length):
                return rule.action, rule
        return "keep", None


def build_rules(config: dict | None, columns: Iterable[str]) -> dict[str, ColumnRules]:
    """``{"defaults": {...}, "columns": {name: {...}}}`` -> rules per column."""
    config = config or {}
    unknown = set(config) - {"defaults", "columns"}
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in missing-value config; use 'defaults' and 'columns'")
    defaults = ColumnRules.from_dict(config.get("defaults"))
    per_column = config.get("columns") or {}
    return {col: ColumnRules.from_dict(per_column.get(col), defaults) for col in columns}


# ---------------------------------------------------------------- the engine
@dataclass
class ColumnPlan:
    """What the rules would do to one column: one row per run."""

    action: np.ndarray  # per position: action name or "" (not missing)
    runs: pd.DataFrame  # kind, start_pos, end_pos, length, action, rule, n_nan, n_zeros


def prepare_values(series: pd.Series, rules: ColumnRules) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float).copy()
    if rules.nan_to_zero:
        values[~np.isfinite(values)] = 0.0
    return values


def plan_column(values: np.ndarray, rules: ColumnRules) -> ColumnPlan:
    """Classify every run of missing values of one column."""
    n = values.size
    is_nan = ~np.isfinite(values)
    is_zero = np.isfinite(values) & (values == 0)
    nan_m = is_nan if rules.nan else np.zeros(n, dtype=bool)
    zero_m = is_zero if rules.zeros else np.zeros(n, dtype=bool)
    if rules.merge:
        masks = [("missing", nan_m | zero_m)]
    else:
        masks = [("nan", nan_m), ("zero", zero_m)]
    action = np.full(n, "", dtype=object)
    rows = []
    for kind, mask in masks:
        for start, end in runs_of(mask):
            length = end - start + 1
            act, rule = rules.action_for(kind, length)
            action[start:end + 1] = act
            rows.append({
                "kind": kind, "start_pos": start, "end_pos": end, "length": length, "action": act,
                "rule": rule.describe() if rule else "no rule",
                "n_nan": int(is_nan[start:end + 1].sum()), "n_zeros": int(is_zero[start:end + 1].sum()),
            })
    runs = pd.DataFrame(rows, columns=["kind", "start_pos", "end_pos", "length", "action", "rule",
                                       "n_nan", "n_zeros"])
    return ColumnPlan(action=action, runs=runs)


def interpolate_positions(values: np.ndarray, positions: np.ndarray, method: str = "linear",
                          index: pd.Index | None = None) -> tuple[np.ndarray, int]:
    """Linearly interpolate ``values`` at ``positions`` from the known neighbours.

    Positions that cannot be interpolated (at the edges) become NaN. Returns
    the new values and the number of such failures.
    """
    out = values.copy()
    if not positions.any():
        return out, 0
    work = pd.Series(values.copy(), index=index)
    work[positions] = np.nan
    filled = work.interpolate(method=method, limit_area="inside").to_numpy(dtype=float)
    out[positions] = filled[positions]
    failed = int((positions & ~np.isfinite(out)).sum())
    return out, failed


def apply_fills(values: np.ndarray, action: np.ndarray, extra_interpolate: np.ndarray | None = None,
                method: str = "linear", index: pd.Index | None = None) -> tuple[np.ndarray, dict]:
    """Apply the fill actions of a plan (everything except ``drop_rows``)."""
    interp = action == "interpolate"
    if extra_interpolate is not None:
        interp = interp | extra_interpolate
    out, failed = interpolate_positions(values, interp, method, index)
    stats = {"interpolated": int(interp.sum()) - failed, "interpolation_failed": failed}
    for act in ("fill_zero", "ffill", "bfill", "set_nan"):
        mask = action == act
        stats[act] = int(mask.sum())
        if not mask.any():
            continue
        if act == "fill_zero":
            out[mask] = 0.0
        elif act == "set_nan":
            out[mask] = np.nan
        else:
            work = pd.Series(out.copy())
            work[mask] = np.nan
            work = work.ffill() if act == "ffill" else work.bfill()
            out[mask] = work.to_numpy(dtype=float)[mask]
    return out, stats


def _missing_after(values: np.ndarray, rules: ColumnRules) -> int:
    mask = np.zeros(values.size, dtype=bool)
    if rules.nan:
        mask |= ~np.isfinite(values)
    if rules.zeros:
        mask |= np.isfinite(values) & (values == 0)
    return int(mask.sum())


def drop_masks(df: pd.DataFrame, rules: dict[str, ColumnRules],
               scope: Iterable[str] | None = None) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Rows to drop because of ``drop_rows`` runs.

    Only columns in ``scope`` (default: all columns with rules) can remove a
    row. Returns the row mask, a per-column mask (for "reason" reports) and
    the table of the runs that caused dropping.
    """
    scope = list(rules) if scope is None else [c for c in scope if c in rules]
    per_column = pd.DataFrame(False, index=df.index, columns=scope)
    tables = []
    for col in scope:
        values = prepare_values(df[col], rules[col])
        plan = plan_column(values, rules[col])
        drop = plan.action == "drop_rows"
        per_column[col] = drop
        runs = plan.runs[plan.runs["action"] == "drop_rows"].copy()
        if not runs.empty:
            runs.insert(0, "column", col)
            runs["start"] = df.index[runs["start_pos"].to_numpy()]
            runs["end"] = df.index[runs["end_pos"].to_numpy()]
            tables.append(runs)
    columns = ["column", "kind", "start", "end", "length", "n_nan", "n_zeros"]
    blocks = pd.concat(tables)[columns].reset_index(drop=True) if tables else pd.DataFrame(columns=columns)
    return per_column.any(axis=1), per_column, blocks


def fill_frame(df: pd.DataFrame, rules: dict[str, ColumnRules], extra_interpolate: pd.DataFrame | None = None,
               method: str = "linear") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply fill actions to every column with rules; returns the frame and a per-column report."""
    out = df.copy()
    rows = []
    for col, col_rules in rules.items():
        if col not in out.columns:
            continue
        values = prepare_values(out[col], col_rules)
        plan = plan_column(values, col_rules)
        extra = None
        if extra_interpolate is not None and col in extra_interpolate.columns:
            extra = extra_interpolate[col].to_numpy(dtype=bool)
        new_values, stats = apply_fills(values, plan.action, extra, method, out.index if method == "time" else None)
        n = max(len(values), 1)
        missing_before = int((plan.action != "").sum())
        missing_after = _missing_after(new_values, col_rules)
        rows.append({
            "column": col,
            "missing_before": missing_before,
            "runs": len(plan.runs),
            "kept": int((plan.action == "keep").sum()),
            "outliers_interpolated": int(extra.sum()) if extra is not None else 0,
            **stats,
            "missing_after": missing_after,
            "missing_before_pct": round(100 * missing_before / n, 4),
            "missing_after_pct": round(100 * missing_after / n, 4),
        })
        out[col] = new_values
    return out, pd.DataFrame(rows).set_index("column") if rows else pd.DataFrame()


def plan_table(df: pd.DataFrame, rules: dict[str, ColumnRules]) -> pd.DataFrame:
    """Dry run: for every column and rule, how many runs/points would be affected."""
    rows = []
    for col, col_rules in rules.items():
        if col not in df.columns:
            continue
        plan = plan_column(prepare_values(df[col], col_rules), col_rules)
        if plan.runs.empty:
            continue
        grouped = plan.runs.groupby(["kind", "rule", "action"], sort=False)
        for (kind, rule, action), group in grouped:
            rows.append({"column": col, "kind": kind, "length": rule, "action": action,
                         "runs": len(group), "points": int(group["length"].sum())})
    return pd.DataFrame(rows, columns=["column", "kind", "length", "action", "runs", "points"])


# ------------------------------------------------------------ transformers
class MissingValueProcessor(BaseTransformer):
    """Apply run-length rules per column: drop long runs, then fill the rest.

    >>> MissingValueProcessor({
    ...     "defaults": {"zeros": True, "runs": {"keep": [1, 3], "interpolate": [4, 24],
    ...                                          "drop_rows": [25, None]}},
    ...     "columns": {"T_AIR": {"zeros": False}},
    ... })

    ``columns`` limits which columns are processed (default: all numeric);
    ``drop_scope`` limits which columns may remove rows (default: processed).
    """

    changes_rows = True

    def __init__(self, rules: dict | None = None, columns: list[str] | None = None,
                 drop_scope: list[str] | None = None, method: str = "linear"):
        self.rules = rules
        self.columns = columns
        self.drop_scope = drop_scope
        self.method = method

    def _column_rules(self, df: pd.DataFrame) -> dict[str, ColumnRules]:
        cols = self.columns if self.columns is not None else [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        return build_rules(self.rules, [c for c in cols if c in df.columns])

    def plan(self, data) -> pd.DataFrame:
        """What the rules would do, without changing the data."""
        df = as_frame(data)
        return plan_table(df, self._column_rules(df))

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        rules = self._column_rules(df)
        rows_before = len(df)
        drop, per_column, blocks = drop_masks(df, rules, self.drop_scope)
        periods = removed_periods(drop, per_column)
        df = df.loc[~drop.to_numpy()]
        df, fills = fill_frame(df, rules, method=self.method)
        self.report_ = {
            "rows_before": rows_before,
            "rows_dropped": int(drop.sum()),
            "rows_after": len(df),
            "blocks": blocks,
            "removed_periods": periods,
            "fills": fills,
        }
        return df


class DeleteLargeGaps(MissingValueProcessor):
    """Drop rows inside runs of missing values of ``min_gap`` or more."""

    def __init__(self, min_gap: int = 25, columns: list[str] | None = None, zeros: bool = False,
                 merge: bool = True):
        self.min_gap = min_gap
        self.zeros = zeros
        self.merge = merge
        super().__init__(
            {"defaults": {"zeros": zeros, "merge": merge, "runs": {"drop_rows": [min_gap, None]}}},
            columns=columns,
        )


class LinearInterpolation(MissingValueProcessor):
    """Linearly interpolate runs of ``min_gap..max_gap`` missing values."""

    changes_rows = False

    def __init__(self, max_gap: int | None = 3, min_gap: int = 1, columns: list[str] | None = None,
                 zeros: bool = False, merge: bool = True, method: str = "linear"):
        self.max_gap = max_gap
        self.min_gap = min_gap
        self.zeros = zeros
        self.merge = merge
        super().__init__(
            {"defaults": {"zeros": zeros, "merge": merge, "runs": {"interpolate": [min_gap, max_gap]}}},
            columns=columns, method=method,
        )


class ForwardFill(MissingValueProcessor):
    """Fill runs of up to ``max_gap`` missing values with the previous value."""

    changes_rows = False

    def __init__(self, max_gap: int | None = 2, columns: list[str] | None = None, zeros: bool = False):
        self.max_gap = max_gap
        self.zeros = zeros
        super().__init__({"defaults": {"zeros": zeros, "runs": {"ffill": [1, max_gap]}}}, columns=columns)


class NanToZero(BaseTransformer):
    """Replace NaN by 0 (e.g. when a missing value means "no production")."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = self.columns if self.columns is not None else list(df.columns)
        counts = {}
        for col in cols:
            counts[col] = int(df[col].isna().sum())
            df[col] = df[col].fillna(0.0)
        self.report_ = {"replaced": counts}
        return df


class MedianImputer(BaseTransformer):
    """Fill NaN with the median learned on the training data, optionally per group.

    >>> MedianImputer(columns=["T_AIR"], groupby=["month", "hour"])   # calendar groups
    """

    stateless = False
    _CALENDAR = {"hour": "hour", "month": "month", "weekday": "weekday", "day_of_year": "dayofyear"}

    def __init__(self, columns: list[str] | None = None, groupby: list[str] | None = None):
        self.columns = columns
        self.groupby = groupby

    def _keys(self, df: pd.DataFrame) -> list:
        keys = []
        for key in self.groupby or []:
            if key in df.columns:
                keys.append(df[key].to_numpy())
            elif key in self._CALENDAR:
                keys.append(getattr(df.index, self._CALENDAR[key]))
            else:
                raise KeyError(f"Group key {key!r} is neither a column nor one of {list(self._CALENDAR)}")
        return keys

    def _fit(self, df: pd.DataFrame) -> None:
        cols = self.columns if self.columns is not None else [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.columns_ = cols
        self.global_median_ = df[cols].median()
        self.medians_ = df[cols].groupby(self._keys(df)).median() if self.groupby else None

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        filled = {}
        for col in self.columns_:
            missing = df[col].isna()
            filled[col] = int(missing.sum())
            if not missing.any():
                continue
            if self.medians_ is not None:
                keys = pd.MultiIndex.from_arrays(self._keys(df)) if len(self.groupby) > 1 \
                    else pd.Index(self._keys(df)[0])
                values = self.medians_[col].reindex(keys).to_numpy()
                values = np.where(np.isfinite(values), values, self.global_median_[col])
            else:
                values = np.full(len(df), self.global_median_[col])
            df.loc[missing, col] = values[missing.to_numpy()]
        self.report_ = {"filled": filled}
        return df
