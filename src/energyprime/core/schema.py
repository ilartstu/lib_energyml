"""Station specification: what every column is and how to treat it.

One YAML file per station::

    name: SPP_2
    capacity: 50            # MW; null -> metrics are normalised by the maximum
    time: {column: TIMESTAMP, format: "%d.%m.%y %H:%M", freq: 1h}
    columns:
      POWER:     {role: target, unit: MW, limits: [-50, null]}
      T_AIR:     {role: feature, available: forecast, limits: [-50, 60]}
      T_PANELS:  {role: feature, available: history}
      DIRECTION: {role: feature, cyclic: 360}
      OTHER:     ignore
    cleaning: {...}         # see energyprime.preprocessing.Cleaner

* ``role`` — ``target`` / ``feature`` / ``ignore``. Columns not listed are ignored.
* ``available`` — ``forecast`` (known for the forecast horizon, e.g. from NWP),
  ``history`` (measured only, e.g. panel temperature) or ``always`` (calendar).
* ``limits`` — physical range; values outside are treated as outliers.
* ``cyclic`` — period for sin/cos encoding (360 for a direction).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROLES = ("target", "feature", "ignore")
AVAILABILITY = ("forecast", "history", "always")


def _check_keys(data: dict, allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in {where}; allowed: {sorted(allowed)}"
        )


def _limits(value: Any) -> tuple[float | None, float | None]:
    if value is None:
        return (None, None)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"limits must be [low, high] (null for no bound), got {value!r}")
    low, high = value
    return (None if low is None else float(low), None if high is None else float(high))


@dataclass
class ColumnSpec:
    name: str
    role: str = "feature"
    available: str | None = None
    unit: str | None = None
    label: str | None = None
    limits: tuple[float | None, float | None] = (None, None)
    cyclic: float | None = None

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Column {self.name!r}: role must be one of {ROLES}, got {self.role!r}")
        if self.available is None:
            self.available = "history" if self.role == "target" else "forecast"
        if self.available not in AVAILABILITY:
            raise ValueError(
                f"Column {self.name!r}: available must be one of {AVAILABILITY}, got {self.available!r}"
            )
        self.limits = _limits(self.limits)
        if self.cyclic is not None:
            self.cyclic = float(self.cyclic)

    @classmethod
    def from_value(cls, name: str, value: Any) -> "ColumnSpec":
        if value is None:
            return cls(name=name)
        if isinstance(value, str):
            return cls(name=name, role=value)
        if not isinstance(value, dict):
            raise ValueError(f"Column {name!r}: expected a role string or a mapping, got {value!r}")
        allowed = {f.name for f in fields(cls)} - {"name"}
        _check_keys(value, allowed, f"column {name!r}")
        return cls(name=name, **value)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"role": self.role}
        default_available = "history" if self.role == "target" else "forecast"
        if self.available != default_available:
            out["available"] = self.available
        for key in ("unit", "label", "cyclic"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if self.limits != (None, None):
            out["limits"] = list(self.limits)
        return out


@dataclass
class TimeSpec:
    column: str | None = None
    format: str | None = None
    freq: str | None = None
    timezone: str | None = None
    dayfirst: bool = False


@dataclass
class ReadSpec:
    sep: str = "auto"
    header_row: int | str = "auto"
    skip_after: int = 0
    sheet: str | int | None = None
    encoding: str = "auto"
    rename: dict[str, str] = field(default_factory=dict)


@dataclass
class StationSpec:
    name: str = "station"
    capacity: float | None = None
    description: str | None = None
    time: TimeSpec = field(default_factory=TimeSpec)
    read: ReadSpec = field(default_factory=ReadSpec)
    columns: dict[str, ColumnSpec] = field(default_factory=dict)
    cleaning: dict = field(default_factory=dict)
    series_column: str | None = None

    # ------------------------------------------------------------ properties
    @property
    def target(self) -> str | None:
        targets = [n for n, c in self.columns.items() if c.role == "target"]
        if len(targets) > 1:
            raise ValueError(f"Station {self.name!r} has several targets: {targets}")
        return targets[0] if targets else None

    @property
    def features(self) -> list[str]:
        return [n for n, c in self.columns.items() if c.role == "feature"]

    @property
    def used_columns(self) -> list[str]:
        target = self.target
        return ([target] if target else []) + self.features

    @property
    def limits(self) -> dict[str, tuple[float | None, float | None]]:
        return {n: c.limits for n, c in self.columns.items() if c.limits != (None, None)}

    @property
    def cyclic_columns(self) -> dict[str, float]:
        return {n: c.cyclic for n, c in self.columns.items() if c.cyclic is not None}

    def column(self, name: str) -> ColumnSpec | None:
        return self.columns.get(name)

    def availability(self, name: str) -> str | None:
        column = self.columns.get(name)
        return column.available if column else None

    def label(self, name: str) -> str:
        column = self.columns.get(name)
        return column.label if column and column.label else name

    # --------------------------------------------------------------- building
    @classmethod
    def from_dict(cls, data: dict) -> "StationSpec":
        data = copy.deepcopy(data or {})
        _check_keys(data, {f.name for f in fields(cls)}, "station spec")
        time = data.pop("time", {}) or {}
        read = data.pop("read", {}) or {}
        _check_keys(time, {f.name for f in fields(TimeSpec)}, "'time'")
        _check_keys(read, {f.name for f in fields(ReadSpec)}, "'read'")
        columns = {
            str(name): ColumnSpec.from_value(str(name), value)
            for name, value in (data.pop("columns", {}) or {}).items()
        }
        spec = cls(time=TimeSpec(**time), read=ReadSpec(**read), columns=columns, **data)
        _ = spec.target  # validates: at most one target
        return spec

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StationSpec":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def load(cls, spec: "StationSpec | dict | str | Path | None") -> "StationSpec":
        """Accept a StationSpec, a dict or a path to a YAML file."""
        if spec is None:
            return cls()
        if isinstance(spec, StationSpec):
            return spec
        if isinstance(spec, dict):
            return cls.from_dict(spec)
        return cls.from_yaml(spec)

    @classmethod
    def infer(cls, df: pd.DataFrame, target: str | None = None, name: str = "station",
              time_column: str | None = None) -> "StationSpec":
        """Spec for a table without a YAML file: numeric columns become features."""
        columns = {}
        for col in df.columns:
            if col == time_column:
                continue
            if col == target:
                columns[col] = ColumnSpec(col, role="target")
            elif pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col]):
                columns[col] = ColumnSpec(col, role="feature")
            else:
                columns[col] = ColumnSpec(col, role="ignore")
        if target is not None and target not in columns:
            raise KeyError(f"Target column {target!r} not found; columns: {list(df.columns)}")
        return cls(name=name, time=TimeSpec(column=time_column), columns=columns)

    # ---------------------------------------------------------------- export
    def to_dict(self) -> dict:
        out: dict[str, Any] = {"name": self.name}
        if self.capacity is not None:
            out["capacity"] = self.capacity
        if self.description:
            out["description"] = self.description
        time = {k: v for k, v in vars(self.time).items() if v not in (None, False)}
        if time:
            out["time"] = time
        read_defaults = vars(ReadSpec())
        read = {k: v for k, v in vars(self.read).items() if v != read_defaults[k]}
        if read:
            out["read"] = read
        if self.series_column:
            out["series_column"] = self.series_column
        out["columns"] = {name: c.to_dict() for name, c in self.columns.items()}
        if self.cleaning:
            out["cleaning"] = copy.deepcopy(self.cleaning)
        return out

    def to_yaml(self, path: str | Path | None = None) -> str:
        text = yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def copy(self) -> "StationSpec":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (f"StationSpec(name={self.name!r}, target={self.target!r}, "
                f"features={self.features}, capacity={self.capacity})")
