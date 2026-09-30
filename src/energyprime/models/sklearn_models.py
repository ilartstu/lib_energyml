"""Any scikit-learn regressor behind the common interface (baselines for benchmarks)."""
from __future__ import annotations

import inspect
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.base import BaseModel
from ..core.registry import import_object

ESTIMATORS = {
    "linear": "sklearn.linear_model:LinearRegression",
    "ridge": "sklearn.linear_model:Ridge",
    "random_forest": "sklearn.ensemble:RandomForestRegressor",
    "gradient_boosting": "sklearn.ensemble:HistGradientBoostingRegressor",
    "knn": "sklearn.neighbors:KNeighborsRegressor",
}


class SklearnRegressor(BaseModel):
    """>>> SklearnRegressor("random_forest", params={"n_estimators": 300}, seed=42)
    >>> SklearnRegressor(my_sklearn_estimator)
    """

    def __init__(self, estimator="linear", params: dict | None = None, seed: int | None = None):
        self.estimator = estimator
        self.params = params
        self.seed = seed

    def _make(self):
        if isinstance(self.estimator, str):
            if self.estimator not in ESTIMATORS:
                raise KeyError(f"Unknown estimator {self.estimator!r}; available: {sorted(ESTIMATORS)}")
            cls = import_object(ESTIMATORS[self.estimator])
            params = dict(self.params or {})
            if self.seed is not None and "random_state" in inspect.signature(cls).parameters:
                params.setdefault("random_state", self.seed)
            return cls(**params)
        from sklearn.base import clone

        estimator = clone(self.estimator)
        if self.params:
            estimator.set_params(**self.params)
        return estimator

    def fit(self, X, y, X_val=None, y_val=None) -> "SklearnRegressor":
        self.estimator_ = self._make().fit(X, np.asarray(y, dtype=float).ravel())
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.fitted_ = True
        return self

    def predict(self, X, weights: str = "best") -> np.ndarray:
        self._check_fitted()
        self._check_weights(weights)
        return np.asarray(self.estimator_.predict(X), dtype=float).ravel()

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        if hasattr(self.estimator_, "feature_importances_"):
            values = self.estimator_.feature_importances_
        elif hasattr(self.estimator_, "coef_"):
            values = np.abs(np.ravel(self.estimator_.coef_))
        else:
            raise NotImplementedError(f"{type(self.estimator_).__name__} has no feature importance")
        names = self.feature_names_ or [f"x{i}" for i in range(len(values))]
        return pd.Series(values, index=names).sort_values(ascending=False)

    def _save_state(self, path: Path) -> None:
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def _from_saved(cls, path: Path, meta: dict) -> "SklearnRegressor":
        with open(path / "model.pkl", "rb") as f:
            return pickle.load(f)
