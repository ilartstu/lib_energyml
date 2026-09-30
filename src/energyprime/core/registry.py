"""Name -> object registries (models, recipes, outlier detectors, metrics).

A registry lets configs and the GUI refer to components by name
(``"xgboost"``, ``"zscore"``) and list what is available. Entries can be lazy
(``"package.module:Name"``) so that heavy frameworks are imported only when a
component is actually requested.
"""
from __future__ import annotations

import importlib
from typing import Any


def import_object(path: str) -> Any:
    """Import ``"package.module:attr"`` and return the attribute."""
    module_name, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"Expected 'module:attr', got {path!r}")
    return getattr(importlib.import_module(module_name), attr)


class _Lazy:
    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str, obj: Any = None, *, lazy: str | None = None,
                 overwrite: bool = False) -> Any:
        """Register ``obj`` (or a lazy import path) under ``name``.

        Without ``obj`` and ``lazy`` works as a decorator.
        """
        if obj is None and lazy is None:
            def decorator(target: Any) -> Any:
                self.register(name, target, overwrite=overwrite)
                return target
            return decorator
        key = name.lower()
        if key in self._items and not overwrite:
            raise KeyError(f"{self.kind} {name!r} is already registered")
        self._items[key] = _Lazy(lazy) if lazy is not None else obj
        return obj

    def get(self, name: str) -> Any:
        key = name.lower()
        if key not in self._items:
            raise KeyError(f"Unknown {self.kind} {name!r}. Available: {', '.join(self.names())}")
        item = self._items[key]
        if isinstance(item, _Lazy):
            item = import_object(item.path)
            self._items[key] = item
        return item

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._items

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}: {', '.join(self.names())})"
