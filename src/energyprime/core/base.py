"""Base contracts: parameters, transformers and models.

Conventions (as in scikit-learn):

* every ``__init__`` argument is stored as an attribute with the same name;
* everything learned from data ends with an underscore (``mean_``, ``history_``);
* ``clone()`` returns an unfitted copy with the same parameters.
"""
from __future__ import annotations

import copy
import inspect
import json
import pickle
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..utils.validation import as_frame
from .history import TrainingHistory
from .registry import import_object

WEIGHTS = ("best", "last")


class NotFittedError(RuntimeError):
    """Raised when a component is used before ``fit``."""


def _values_equal(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # arrays, frames...
        return a is b


class Params:
    """``get_params`` / ``set_params`` / ``clone`` from the ``__init__`` signature."""

    def get_params(self) -> dict[str, Any]:
        params = {}
        for name, p in inspect.signature(type(self).__init__).parameters.items():
            if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            params[name] = getattr(self, name)
        return params

    def set_params(self, **params: Any) -> "Params":
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"{type(self).__name__} has no parameter {key!r}")
            setattr(self, key, value)
        return self

    def clone(self) -> "Params":
        return type(self)(**copy.deepcopy(self.get_params()))

    def __repr__(self) -> str:
        signature = inspect.signature(type(self).__init__)
        parts = []
        for name, value in self.get_params().items():
            default = signature.parameters[name].default
            if default is inspect.Parameter.empty or not _values_equal(value, default):
                parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


def _is_split(data: Any) -> bool:
    return bool(getattr(data, "__energyprime_split__", False))


def _rewrap(original: Any, frame: pd.DataFrame) -> Any:
    """Return a Dataset if a Dataset came in, otherwise the DataFrame."""
    if hasattr(original, "with_data") and not isinstance(original, pd.DataFrame):
        return original.with_data(frame)
    return frame


class BaseTransformer(Params):
    """DataFrame -> DataFrame step with ``fit / transform / fit_transform``.

    Accepts a DataFrame, a Dataset (returns a Dataset) or a Split (fits on
    ``split.train``, transforms every part). The input is never modified.
    After ``transform`` the step's report is available via ``get_report()``;
    after transforming a Split it is ``{"train": ..., "val": ..., "test": ...}``.
    """

    #: learns nothing in ``fit`` (can be used without fitting)
    stateless: ClassVar[bool] = True
    #: may remove rows (only allowed while cleaning historical data)
    changes_rows: ClassVar[bool] = False

    def fit(self, data: Any, y: Any = None) -> "BaseTransformer":
        if _is_split(data):
            data = data.train
        self._fit(as_frame(data))
        self.fitted_ = True
        return self

    def transform(self, data: Any) -> Any:
        if _is_split(data):
            outputs, reports = {}, {}
            for name, part in data.parts().items():
                outputs[name] = self.transform(part)
                reports[name] = self.get_report()
            self.report_ = reports
            return data.replace(**outputs)
        if not self.stateless and not getattr(self, "fitted_", False):
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first")
        return _rewrap(data, self._transform(as_frame(data).copy()))

    def fit_transform(self, data: Any, y: Any = None) -> Any:
        self.fit(data)
        return self.transform(data)

    def get_report(self) -> dict:
        """What the last ``transform`` did (counts, tables)."""
        return getattr(self, "report_", {})

    def _fit(self, df: pd.DataFrame) -> None:
        pass

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class BaseModel(Params):
    """Unified model interface.

    ``fit(X, y, X_val, y_val)`` trains (validation data drives early stopping /
    best-epoch selection), ``predict(X, weights="best"|"last")`` returns a 1-D
    array in the model's own target space (the Forecaster handles scaling).
    """

    #: "table" -> X is 2-D (rows x features); "sequence" -> X is 3-D windows
    input_kind: ClassVar[str] = "table"
    #: provides predict_distribution (mean, std)
    supports_distribution: ClassVar[bool] = False

    def fit(self, X: Any, y: Any, X_val: Any = None, y_val: Any = None) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X: Any, weights: str = "best") -> np.ndarray:
        raise NotImplementedError

    def predict_distribution(self, X: Any, weights: str = "best") -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError(f"{type(self).__name__} does not provide predictive uncertainty")

    def feature_importance(self) -> pd.Series:
        raise NotImplementedError(f"{type(self).__name__} does not provide feature importance")

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "fitted_", False)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError(f"{type(self).__name__} is not fitted; call fit() first")

    @staticmethod
    def _check_weights(weights: str) -> None:
        if weights not in WEIGHTS:
            raise ValueError(f"weights must be one of {WEIGHTS}, got {weights!r}")

    # ------------------------------------------------------------------ saving
    def save(self, path: str | Path) -> Path:
        """Save to a directory: ``model.json`` (class + params), history, state."""
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "class": f"{type(self).__module__}:{type(self).__qualname__}",
            "params": self.get_params(),
        }
        (path / "model.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        history = getattr(self, "history_", None)
        if history is not None:
            (path / "history.json").write_text(json.dumps(history.to_dict(), indent=2))
        self._save_state(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BaseModel":
        return load_model(path)

    @classmethod
    def _from_saved(cls, path: Path, meta: dict) -> "BaseModel":
        model = cls(**meta["params"])
        model._load_state(path)
        history_file = path / "history.json"
        if history_file.exists():
            model.history_ = TrainingHistory.from_dict(json.loads(history_file.read_text()))
        model.fitted_ = True
        return model

    def _save_state(self, path: Path) -> None:
        state = {k: v for k, v in vars(self).items() if k.endswith("_") and k != "history_"}
        with open(path / "state.pkl", "wb") as f:
            pickle.dump(state, f)

    def _load_state(self, path: Path) -> None:
        with open(path / "state.pkl", "rb") as f:
            vars(self).update(pickle.load(f))


def load_model(path: str | Path) -> BaseModel:
    """Load any model saved with ``BaseModel.save``."""
    path = Path(path)
    meta = json.loads((path / "model.json").read_text())
    cls = import_object(meta["class"])
    return cls._from_saved(path, meta)


def _json_default(value: Any) -> Any:
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)
