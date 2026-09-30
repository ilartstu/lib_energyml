"""Forecaster: features + scaling + target transform + model as one object.

Everything that must happen identically in training and in a real forecast
lives here (the GUI loads a saved Forecaster and calls ``predict``). Cleaning
of historical data is *not* part of it — that is :class:`~energyprime.preprocessing.Cleaner`.

>>> fc = Forecaster.from_recipe("xgboost")
>>> fc.fit(split.train, validation=split.val)
>>> pred = fc.predict(split.test, history=split.train)
>>> fc.save("models/xgb_site1"); fc = Forecaster.load("models/xgb_site1")
"""
from __future__ import annotations

import copy
import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core.base import BaseModel, NotFittedError, load_model
from .core.schema import StationSpec
from .evaluation.metrics import evaluate_regression
from .features.calendar import CALENDAR, CalendarFeatures
from .features.cyclic import CyclicEncoder
from .features.lags import LagFeatures, RollingFeatures
from .features.windows import make_windows
from .preprocessing.scaling import ScalingProcessor
from .preprocessing.target import make_target_transform
from .utils.validation import as_frame, check_time_index, infer_freq, numeric_columns, to_timedelta


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames) if len(frames) > 1 else frames[0]


class Forecaster:
    """Turn weather (and optionally history) into a power forecast.

    Parameters
    ----------
    model : BaseModel or str
        A model instance or a registry name (``"xgboost"``, ``"lstm"``, ``"svgp"``...).
    target, features : str, list of str
        Override the station spec (default: ``role: target`` / ``role: feature``).
    calendar : list of str
        Calendar columns to add: ``["year"]`` -> ``YEAR``.
    cyclic : dict
        Sin/cos encoding: ``{"hour": 24, "day_of_year": 365.25}``. Columns with
        ``cyclic:`` in the spec (e.g. ``DIRECTION: 360``) are added automatically.
    lags, rolling : dict
        History as columns: ``lags={"POWER": [1, 24]}``,
        ``rolling={"POWER": {"windows": [24], "stats": ["mean", "std"]}}``.
        Lags of the target make the forecast recursive (see ``predict_recursive``).
    drop : list of str
        Columns removed after feature engineering.
    feature_scaling : str or dict
        ``"minmax"``, ``"standard"``, ``"robust"``, ``"maxabs"`` or
        ``{"method": "maxabs", "exclude": [...], "exclude_calendar_cyclic": True}``.
    target_transform : str or TargetTransform
        ``"identity"`` (default), ``"minmax"``, ``"minmax_logit"``.
    lookback : int
        Window length for sequence models (LSTM).
    interval_k : float
        Default half-width of intervals in std units (``mean ± k*std``).
    """

    def __init__(self, model: BaseModel | str, *, target: str | None = None, features: list[str] | None = None,
                 calendar: list[str] | None = None, cyclic: dict[str, float] | None = None,
                 lags: dict[str, list[int]] | None = None, rolling: dict[str, dict] | None = None,
                 drop: list[str] | None = None, feature_scaling: str | dict | None = None,
                 target_transform: Any = None, lookback: int | None = None, freq: str | None = None,
                 check_gaps: bool = True, use_spec_cyclic: bool = True, interval_k: float = 1.0,
                 spec: StationSpec | dict | str | None = None):
        if isinstance(model, str):
            from .models import get_model

            model = get_model(model)
        self.model = model
        self._params = copy.deepcopy(dict(
            target=target, features=features, calendar=calendar, cyclic=cyclic, lags=lags, rolling=rolling,
            drop=drop, feature_scaling=feature_scaling, target_transform=target_transform, lookback=lookback,
            freq=freq, check_gaps=check_gaps, use_spec_cyclic=use_spec_cyclic, interval_k=interval_k, spec=spec,
        ))
        for key, value in self._params.items():
            setattr(self, key, value)
        self.recipe = None

    # ------------------------------------------------------------ building
    @classmethod
    def from_recipe(cls, name: str, *, model_params: dict | None = None,
                    spec: StationSpec | dict | str | None = None, **overrides: Any) -> "Forecaster":
        """A Forecaster configured as in a recipe (``"xgboost"``, ``"lstm"``, ``"svgp"``...)."""
        from .models import get_model, get_recipe

        recipe = get_recipe(name)
        model = get_model(recipe.model, **{**recipe.model_params, **(model_params or {})})
        forecaster = cls(model, spec=spec, **{**recipe.forecaster, **overrides})
        forecaster.recipe = name
        return forecaster

    def clone(self) -> "Forecaster":
        """Unfitted copy with the same configuration."""
        twin = Forecaster(self.model.clone(), **copy.deepcopy(self._params))
        twin.recipe = self.recipe
        return twin

    @property
    def is_fitted(self) -> bool:
        return getattr(self, "fitted_", False)

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise NotFittedError("Forecaster is not fitted; call fit() first")

    # --------------------------------------------------------- data access
    def _resolve(self, data: Any) -> tuple[pd.DataFrame, StationSpec | None]:
        df = as_frame(data)
        check_time_index(df)
        spec = getattr(data, "spec", None) if not isinstance(data, pd.DataFrame) else None
        if spec is None:
            spec = getattr(self, "spec_", None)
        if spec is None and self.spec is not None:
            spec = StationSpec.load(self.spec)
        return df, spec

    def resolve_target(self, data: Any) -> str:
        if self.target is not None:
            return self.target
        _, spec = self._resolve(data)
        if spec is None or spec.target is None:
            raise ValueError("No target: pass target=... or use a Dataset whose spec has a target")
        return spec.target

    def _input_columns(self, df: pd.DataFrame, spec: StationSpec | None) -> list[str]:
        if self.features is not None:
            cols = list(self.features)
        elif spec is not None and spec.features:
            cols = [c for c in spec.features if c in df.columns]
        else:
            cols = numeric_columns(df)
        cols = [c for c in cols if c != self.target_]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"Feature columns not found in the data: {missing}")
        return cols

    def _build_steps(self, spec: StationSpec | None) -> list:
        steps: list = []
        if self.calendar:
            steps.append(CalendarFeatures(list(self.calendar)))
        periods = dict(self.cyclic or {})
        if self.use_spec_cyclic and spec is not None:
            for col, period in spec.cyclic_columns.items():
                if col in self.input_columns_ and col not in periods:
                    periods[col] = period
        if periods:
            steps.append(CyclicEncoder(periods))
        for col, lags in (self.lags or {}).items():
            steps.append(LagFeatures([col], list(lags), freq=self.freq))
        for col, cfg in (self.rolling or {}).items():
            steps.append(RollingFeatures([col], freq=self.freq, **cfg))
        return steps

    def _context_steps(self) -> int:
        steps = max([s.max_steps() for s in self.steps_ if hasattr(s, "max_steps")] or [0])
        return steps + (self.lookback - 1 if self.lookback else 0)

    def _uses_target_history(self) -> bool:
        return self.target_ in (self.lags or {}) or self.target_ in (self.rolling or {})

    # ------------------------------------------------- feature engineering
    def _engineer(self, df: pd.DataFrame, history: Any = None) -> tuple[pd.DataFrame, np.ndarray]:
        """Engineered features for ``df`` rows, with history rows prepended as context."""
        cols = self.needed_columns_
        frame = df.reindex(columns=cols)
        own = np.ones(len(frame), dtype=bool)
        context = self._context_steps()
        if history is not None and context > 0 and len(frame):
            hist = as_frame(history)
            hist = hist.loc[hist.index < frame.index[0]]
            if self.freq_ is not None:
                hist = hist.loc[hist.index >= frame.index[0] - (context + 1) * self.freq_]
            else:
                hist = hist.iloc[-(context + 1):]
            if len(hist):
                frame = pd.concat([hist.reindex(columns=cols), frame])
                own = np.concatenate([np.zeros(len(hist), dtype=bool), own])
        for step in self.steps_:
            frame = step.transform(frame)
        return frame, own

    def _make_scaler(self) -> ScalingProcessor | None:
        config = self.feature_scaling
        if config is None:
            return None
        config = {"method": config} if isinstance(config, str) else dict(config)
        exclude = list(config.pop("exclude", None) or [])
        if config.pop("exclude_calendar_cyclic", False):
            for step in self.steps_:
                if isinstance(step, CyclicEncoder):
                    for name in step.periods:
                        if name in CALENDAR and name not in self.input_columns_:
                            exclude += list(step.output_names(name))
        return ScalingProcessor(columns=list(self.feature_names_), exclude=exclude, **config)

    def _design(self, df: pd.DataFrame, history: Any, need_target: bool):
        """Model input for the rows of ``df``: (X, timestamps, target values)."""
        eng, own = self._engineer(df, history)
        features = eng[self.feature_names_]
        if self.scaler_ is not None:
            features = self.scaler_.transform(features)
        target = eng[self.target_].to_numpy(dtype=float) if self.target_ in eng else np.full(len(eng), np.nan)
        values = features.to_numpy(dtype=float)
        if self.lookback is None:
            ok = own & np.isfinite(values).all(axis=1)
            if need_target:
                ok &= np.isfinite(target)
            return features.loc[ok], features.index[ok], target[ok]
        ends = own & np.isfinite(target) if need_target else own
        X, positions = make_windows(values, self.lookback, index=features.index, freq=self.freq_,
                                    check_gaps=self.check_gaps, ends=ends)
        return X, features.index[positions], target[positions]

    # ------------------------------------------------------------ training
    def fit(self, data: Any, validation: Any = None, history: Any = None) -> "Forecaster":
        """Fit on ``data`` (Dataset or DataFrame); ``validation`` drives early stopping."""
        df, spec = self._resolve(data)
        self.spec_ = spec
        self.target_ = self.resolve_target(data)
        if self.target_ not in df.columns:
            raise KeyError(f"Target column {self.target_!r} not found")
        kind = getattr(self.model, "input_kind", "table")
        if kind == "sequence" and not self.lookback:
            raise ValueError(f"{type(self.model).__name__} needs windows: pass lookback=...")
        if kind == "table" and self.lookback:
            raise ValueError("lookback (windows) is for sequence models; table models take lags=...")

        self.input_columns_ = self._input_columns(df, spec)
        spec_freq = to_timedelta(spec.time.freq) if spec is not None and spec.time.freq else None
        self.freq_ = to_timedelta(self.freq) or spec_freq or infer_freq(df.index)
        self.steps_ = self._build_steps(spec)
        sources = list((self.lags or {})) + list((self.rolling or {}))
        self.needed_columns_ = list(dict.fromkeys(self.input_columns_ + [self.target_] + sources))

        eng, own = self._engineer(df, history)
        excluded = {self.target_} | set(self.drop or []) | {c for c in sources if c not in self.input_columns_}
        self.feature_names_ = [c for c in eng.columns if c not in excluded]
        self.scaler_ = self._make_scaler()
        if self.scaler_ is not None:
            self.scaler_.fit(eng.loc[own, self.feature_names_])
        y_train = eng.loc[own, self.target_].to_numpy(dtype=float)
        self.target_transform_ = make_target_transform(self.target_transform).fit(y_train[np.isfinite(y_train)])

        X, _, y = self._design(df, history, need_target=True)
        if len(y) == 0:
            raise ValueError("No complete training rows (all rows have NaN features or target)")
        X_val = y_val = None
        if validation is not None:
            val_df, _ = self._resolve(validation)
            context = _concat([as_frame(history) if history is not None else None, df])
            X_val, _, y_val = self._design(val_df, context, need_target=True)
            y_val = self.target_transform_.transform(y_val)
        self.model.fit(X, self.target_transform_.transform(y), X_val, y_val)

        self.info_ = {
            "train_rows": int(own.sum()),
            "train_samples": int(len(y)),
            "train_rows_skipped": int(own.sum() - len(y)),
            "val_samples": 0 if y_val is None else int(len(y_val)),
            "features": list(self.feature_names_),
        }
        self.fitted_ = True
        self._warn_availability()
        return self

    def _warn_availability(self) -> None:
        if self.spec_ is None:
            return
        measured = [c for c in self.feature_names_ if self.spec_.availability(c) == "history"]
        if measured:
            warnings.warn(
                f"Features {measured} are marked 'available: history' in the station spec: they are "
                f"measured only and will be unknown at a real forecast. Scores assume they are known.",
                UserWarning, stacklevel=3,
            )

    # ---------------------------------------------------------- prediction
    def predict(self, data: Any, history: Any = None, weights: str = "best") -> pd.Series:
        """Point forecast for every row of ``data`` (NaN where inputs are incomplete).

        ``history`` — earlier data (e.g. the training part) used as context for
        lags and LSTM windows at the start of ``data``.
        """
        self._check_fitted()
        df, _ = self._resolve(data)
        X, index, _ = self._design(df, history, need_target=False)
        values = np.empty(0)
        if len(index):
            values = self.target_transform_.inverse_transform(self.model.predict(X, weights=weights))
        return pd.Series(values, index=index, name=self.target_, dtype=float).reindex(df.index)

    def predict_interval(self, data: Any, history: Any = None, k: float | None = None,
                         weights: str = "best") -> pd.DataFrame:
        """``pred``, ``lower``, ``upper`` = inverse(mean ± k*std) and ``std`` = (upper - lower) / 2k."""
        self._check_fitted()
        if not getattr(self.model, "supports_distribution", False):
            raise NotImplementedError(f"{type(self.model).__name__} does not predict uncertainty")
        k = self.interval_k if k is None else k
        df, _ = self._resolve(data)
        X, index, _ = self._design(df, history, need_target=False)
        columns = ["pred", "lower", "upper", "std"]
        if not len(index):
            return pd.DataFrame(np.nan, index=df.index, columns=columns)
        mean, std = self.model.predict_distribution(X, weights=weights)
        inverse = self.target_transform_.inverse_transform
        lower, upper = inverse(mean - k * std), inverse(mean + k * std)
        out = pd.DataFrame({"pred": inverse(mean), "lower": lower, "upper": upper,
                            "std": (upper - lower) / (2 * k)}, index=index)
        return out.reindex(df.index)

    def predict_recursive(self, data: Any, history: Any, weights: str = "best") -> pd.Series:
        """Step-by-step forecast feeding predictions back as target lags.

        Needed when the model uses lags of the target: at a real forecast the
        future target is unknown, so each step uses the previous predictions.
        Without target lags this equals ``predict``.
        """
        self._check_fitted()
        if not self._uses_target_history():
            return self.predict(data, history=history, weights=weights)
        if history is None:
            raise ValueError("history with the target up to the forecast start is required")
        df, _ = self._resolve(data)
        target = self.target_
        known = as_frame(history).reindex(columns=self.needed_columns_)
        if len(df):
            known = known.loc[known.index < df.index[0]]
        future = df.reindex(columns=self.needed_columns_).copy()
        future[target] = np.nan
        keep = self._context_steps() + 1
        values = np.full(len(future), np.nan)
        for i in range(len(future)):
            row = future.iloc[[i]].copy()
            X, index, _ = self._design(row, known, need_target=False)
            if len(index):
                values[i] = float(self.target_transform_.inverse_transform(
                    self.model.predict(X, weights=weights))[0])
                row[target] = values[i]
            known = pd.concat([known.iloc[-keep:], row])
        return pd.Series(values, index=future.index, name=target)

    def evaluate(self, data: Any, history: Any = None, metrics=("nmae", "nrmse", "r2"),
                 norm: float | None = None, weights: str = "best", percent: bool = False) -> dict:
        """Metrics on ``data``. ``norm``: capacity from the spec, else the target maximum in ``data``."""
        df, spec = self._resolve(data)
        y_true = df[self.target_]
        source = "given"
        if norm is None:
            if spec is not None and spec.capacity is not None:
                norm, source = float(spec.capacity), "capacity"
            else:
                norm, source = float(y_true.abs().max()), "max"
        pred = self.predict(data, history=history, weights=weights)
        scores = evaluate_regression(y_true, pred, metrics, norm=norm, percent=percent)
        return {**scores, "norm": norm, "norm_source": source}

    # ----------------------------------------------------------- inspection
    @property
    def history_(self):
        return getattr(self.model, "history_", None)

    def feature_importance(self) -> pd.Series:
        return self.model.feature_importance()

    @property
    def required_inputs(self) -> dict:
        """What a forecast needs: input columns, their availability, history length."""
        self._check_fitted()
        columns = [c for c in self.needed_columns_ if c != self.target_]
        return {
            "columns": columns,
            "availability": {c: (self.spec_.availability(c) if self.spec_ else None) for c in columns},
            "target_history": self._uses_target_history(),
            "history_steps": self._context_steps(),
            "freq": str(self.freq_) if self.freq_ is not None else None,
        }

    # --------------------------------------------------------------- saving
    def save(self, path: str | Path) -> Path:
        """Directory with the model (``model/``), the pipeline state and a readable summary."""
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model")
        state = dict(self.__dict__)
        state["model"] = None
        with open(path / "forecaster.pkl", "wb") as f:
            pickle.dump(state, f)
        summary = {"recipe": self.recipe, "model": type(self.model).__name__, "target": self.target_,
                   "features": self.feature_names_, "lookback": self.lookback,
                   "required_inputs": self.required_inputs}
        (path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Forecaster":
        path = Path(path)
        with open(path / "forecaster.pkl", "rb") as f:
            state = pickle.load(f)
        forecaster = cls.__new__(cls)
        forecaster.__dict__.update(state)
        forecaster.model = load_model(path / "model")
        return forecaster

    def __repr__(self) -> str:
        parts = [f"model={type(self.model).__name__}"]
        if self.recipe:
            parts.append(f"recipe={self.recipe!r}")
        if self.is_fitted:
            parts += [f"target={self.target_!r}", f"features={len(self.feature_names_)}"]
        if self.lookback:
            parts.append(f"lookback={self.lookback}")
        parts.append("fitted" if self.is_fitted else "not fitted")
        return f"Forecaster({', '.join(parts)})"
