"""Recipes: a model plus the data preparation that suits it.

A recipe is what differs between the notebooks: which extra features, how
features and the target are scaled, and whether the model needs windows.
Everything can be overridden::

    Forecaster.from_recipe("svgp")                                   # as in the notebook
    Forecaster.from_recipe("xgboost", lags={"POWER": [24, 48]})      # + lags
    Forecaster.from_recipe("lstm", lookback=48, model_params={"units": 64})
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from ..core.registry import Registry

RECIPES = Registry("recipe")

TIME_OF_DAY_AND_YEAR = {"hour": 24, "day_of_year": 365.25}


@dataclass
class Recipe:
    name: str
    model: str
    model_params: dict = field(default_factory=dict)
    forecaster: dict = field(default_factory=dict)
    description: str = ""


def register_recipe(recipe: Recipe, overwrite: bool = False) -> None:
    RECIPES.register(recipe.name, recipe, overwrite=overwrite)


def get_recipe(name: str) -> Recipe:
    return copy.deepcopy(RECIPES.get(name))


def list_recipes() -> pd.DataFrame:
    rows = [{"name": r.name, "model": r.model, "description": r.description}
            for r in (RECIPES.get(n) for n in RECIPES.names())]
    return pd.DataFrame(rows)


register_recipe(Recipe(
    name="xgboost",
    model="xgboost",
    forecaster={"calendar": ["year"], "cyclic": dict(TIME_OF_DAY_AND_YEAR)},
    description="XGBoost on same-time weather + YEAR + cyclic hour/day of year; no scaling "
                "(trees do not need it). History: optional lags.",
))

register_recipe(Recipe(
    name="lstm",
    model="lstm",
    forecaster={"cyclic": dict(TIME_OF_DAY_AND_YEAR), "feature_scaling": "minmax",
                "target_transform": "minmax", "lookback": 24},
    description="Keras LSTM on 24-hour windows of weather features (no POWER lags); min-max "
                "scaling of features and target from the training part.",
))

register_recipe(Recipe(
    name="svgp",
    model="svgp",
    forecaster={"calendar": ["year"], "cyclic": dict(TIME_OF_DAY_AND_YEAR),
                "feature_scaling": {"method": "maxabs", "exclude_calendar_cyclic": True},
                "target_transform": "minmax_logit"},
    description="Sparse variational GP with MLP mean; features divided by max |x|; target "
                "min-max + logit (keeps forecasts inside the observed range); mean ± std intervals.",
))

register_recipe(Recipe(
    name="linear",
    model="sklearn",
    model_params={"estimator": "linear"},
    forecaster={"cyclic": dict(TIME_OF_DAY_AND_YEAR), "feature_scaling": "standard"},
    description="Linear regression baseline (scikit-learn).",
))

register_recipe(Recipe(
    name="random_forest",
    model="sklearn",
    model_params={"estimator": "random_forest", "params": {"n_estimators": 300, "n_jobs": -1}, "seed": 42},
    forecaster={"calendar": ["year"], "cyclic": dict(TIME_OF_DAY_AND_YEAR)},
    description="Random forest baseline (scikit-learn).",
))
