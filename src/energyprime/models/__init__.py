"""Models behind one interface. Heavy frameworks are imported only on use:
``from energyprime.models import XGBoostRegressor`` imports xgboost, but
``import energyprime`` does not.
"""
from __future__ import annotations

from typing import Any

from ..core.registry import Registry
from .recipes import RECIPES, Recipe, get_recipe, list_recipes, register_recipe

MODELS = Registry("model")
MODELS.register("xgboost", lazy="energyprime.models.xgboost_model:XGBoostRegressor")
MODELS.register("lstm", lazy="energyprime.models.lstm:LSTMRegressor")
MODELS.register("svgp", lazy="energyprime.models.svgp:SVGPRegressor")
MODELS.register("sklearn", lazy="energyprime.models.sklearn_models:SklearnRegressor")

_CLASSES = {
    "XGBoostRegressor": "xgboost",
    "LSTMRegressor": "lstm",
    "SVGPRegressor": "svgp",
    "SklearnRegressor": "sklearn",
}


def get_model(name: str, **params: Any):
    """Create a model by registry name: ``get_model("xgboost", max_depth=8)``."""
    return MODELS.get(name)(**params)


def list_models() -> list[str]:
    return MODELS.names()


def __getattr__(name: str):
    if name in _CLASSES:
        return MODELS.get(_CLASSES[name])
    raise AttributeError(f"module 'energyprime.models' has no attribute {name!r}")


__all__ = ["MODELS", "get_model", "list_models", "Recipe", "RECIPES", "get_recipe", "list_recipes",
           "register_recipe", *_CLASSES]
