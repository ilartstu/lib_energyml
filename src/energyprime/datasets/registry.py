"""Named datasets: register a file + spec once, load it by name later."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset import Dataset
from .io import load_table


@dataclass
class _Entry:
    path: Path
    spec: Any
    description: str | None


_DATASETS: dict[str, _Entry] = {}


def register_dataset(name: str, path: str | Path, spec: Any = None, description: str | None = None,
                     overwrite: bool = False) -> None:
    """Remember where a dataset lives and which spec to load it with."""
    if name in _DATASETS and not overwrite:
        raise KeyError(f"Dataset {name!r} is already registered")
    _DATASETS[name] = _Entry(Path(path), spec, description)


def list_datasets() -> pd.DataFrame:
    rows = [{"name": n, "path": str(e.path), "description": e.description} for n, e in _DATASETS.items()]
    return pd.DataFrame(rows, columns=["name", "path", "description"])


def load_dataset(name: str, **kwargs: Any) -> Dataset:
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset {name!r}. Registered: {sorted(_DATASETS)}")
    entry = _DATASETS[name]
    return load_table(entry.path, spec=entry.spec, name=name, **kwargs)
