"""XGBoost — the XGBoost notebook as a model (``xgb.train`` with early stopping).

Defaults = the notebook: squared error, RMSE, learning rate 0.03, depth 6,
subsample / colsample 0.8, lambda 1, up to 5000 rounds, early stopping after
200 rounds without improvement on the validation set, seed 42.

``predict(X, weights="best")`` uses the trees up to the best iteration (as
the notebook); ``weights="last"`` uses all trained trees.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from ..core.base import BaseModel
from ..core.history import TrainingHistory


class XGBoostRegressor(BaseModel):
    """>>> XGBoostRegressor(n_estimators=5000, learning_rate=0.03, max_depth=6, seed=42)"""

    def __init__(self, n_estimators: int = 5000, learning_rate: float = 0.03, max_depth: int = 6,
                 subsample: float = 0.8, colsample_bytree: float = 0.8, reg_lambda: float = 1.0,
                 objective: str = "reg:squarederror", eval_metric: str = "rmse",
                 early_stopping_rounds: int | None = 200, seed: int = 42, nthread: int = -1,
                 verbose_eval: int | bool = False, params: dict | None = None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.objective = objective
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.seed = seed
        self.nthread = nthread
        self.verbose_eval = verbose_eval
        self.params = params

    def booster_params(self) -> dict:
        params = {"objective": self.objective, "eval_metric": self.eval_metric,
                  "learning_rate": self.learning_rate, "max_depth": self.max_depth,
                  "subsample": self.subsample, "colsample_bytree": self.colsample_bytree,
                  "lambda": self.reg_lambda, "seed": self.seed, "nthread": self.nthread}
        params.update(self.params or {})
        return params

    def fit(self, X, y, X_val=None, y_val=None) -> "XGBoostRegressor":
        dtrain = xgb.DMatrix(X, label=np.asarray(y, dtype=float).ravel())
        evals = [(dtrain, "train")]
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        if has_val:
            evals.append((xgb.DMatrix(X_val, label=np.asarray(y_val, dtype=float).ravel()), "valid"))
        evals_result: dict = {}
        t0 = time.perf_counter()
        booster = xgb.train(self.booster_params(), dtrain, num_boost_round=self.n_estimators, evals=evals,
                            early_stopping_rounds=self.early_stopping_rounds if has_val else None,
                            verbose_eval=self.verbose_eval, evals_result=evals_result)
        train_time = time.perf_counter() - t0

        self.booster_ = booster
        self.best_iteration_ = int(booster.best_iteration) if has_val and self.early_stopping_rounds \
            else booster.num_boosted_rounds() - 1
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        metric = self.eval_metric
        self.history_ = TrainingHistory(
            train=[float(v) for v in evals_result["train"][metric]],
            val=[float(v) for v in evals_result.get("valid", {}).get(metric, [])],
            metric=metric, step_name="round", best_step=self.best_iteration_ + 1,
            train_time_sec=train_time,
            stopped_early=booster.num_boosted_rounds() < self.n_estimators,
        )
        self.fitted_ = True
        return self

    def predict(self, X, weights: str = "best") -> np.ndarray:
        self._check_fitted()
        self._check_weights(weights)
        end = self.best_iteration_ + 1 if weights == "best" else self.booster_.num_boosted_rounds()
        return self.booster_.predict(xgb.DMatrix(X), iteration_range=(0, end))

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        self._check_fitted()
        scores = self.booster_.get_score(importance_type=importance_type)
        names = self.feature_names_ or sorted(scores)
        return pd.Series({n: scores.get(n, 0.0) for n in names}).sort_values(ascending=False)

    def _save_state(self, path: Path) -> None:
        self.booster_.save_model(path / "booster.json")
        (path / "state.json").write_text(json.dumps(
            {"best_iteration": self.best_iteration_, "feature_names": self.feature_names_}))

    def _load_state(self, path: Path) -> None:
        self.booster_ = xgb.Booster()
        self.booster_.load_model(path / "booster.json")
        state = json.loads((path / "state.json").read_text())
        self.best_iteration_ = state["best_iteration"]
        self.feature_names_ = state["feature_names"]
