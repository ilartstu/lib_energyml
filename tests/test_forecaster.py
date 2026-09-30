import warnings

import numpy as np
import pandas as pd
import pytest

from energyprime import Forecaster, split_by_time
from energyprime.evaluation import permutation_importance, r2, shuffled_target_check


@pytest.fixture
def split(solar_dataset):
    return split_by_time(solar_dataset, test="5D", val="5D")


def _fit_quietly(forecaster, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return forecaster.fit(*args, **kwargs)


def test_linear_recipe_end_to_end(split, tmp_path):
    pytest.importorskip("sklearn")
    fc = Forecaster.from_recipe("linear")
    with pytest.warns(UserWarning, match="T_PANELS"):
        fc.fit(split.train, validation=split.val)
    assert {"HOUR_SIN", "DAY_OF_YEAR_COS", "DIRECTION_SIN", "DIRECTION_COS"} <= set(fc.feature_names_)
    assert "DIRECTION" not in fc.feature_names_ and "POWER" not in fc.feature_names_
    pred = fc.predict(split.test)
    assert pred.index.equals(split.test.data.index) and pred.notna().all()
    assert r2(split.test.y, pred) > 0.9
    scores = fc.evaluate(split.test, percent=True)
    assert scores["norm_source"] == "capacity" and scores["norm"] == 50
    assert fc.required_inputs["availability"]["T_PANELS"] == "history"

    fc.save(tmp_path / "fc")
    loaded = Forecaster.load(tmp_path / "fc")
    pd.testing.assert_series_equal(loaded.predict(split.test), pred)
    assert not fc.clone().is_fitted


def test_target_lags_and_recursive_forecast(split):
    pytest.importorskip("sklearn")
    from energyprime.models import SklearnRegressor

    fc = _fit_quietly(Forecaster(SklearnRegressor("linear"), features=["INSOLATION", "T_AIR"],
                                 lags={"POWER": [1, 24]}), split.train)
    history = pd.concat([split.train.data, split.val.data])
    day = split.test.data.iloc[:24]
    one_step = fc.predict(day, history=history)
    recursive = fc.predict_recursive(day, history=history)
    assert one_step.notna().all() and recursive.notna().all()
    assert recursive.iloc[0] == pytest.approx(one_step.iloc[0])
    no_history = fc.predict(day)
    assert no_history.isna().sum() == 24  # lag 24 needs history
    assert fc.required_inputs["target_history"] and fc.required_inputs["history_steps"] == 24


def test_xgboost_recipe_best_and_last(split, tmp_path):
    pytest.importorskip("xgboost")
    fc = _fit_quietly(Forecaster.from_recipe(
        "xgboost", model_params={"n_estimators": 400, "learning_rate": 0.3, "early_stopping_rounds": 10}),
        split.train, validation=split.val)
    history = fc.history_
    assert history.step_name == "round" and history.best_step <= history.last_step
    best, last = fc.predict(split.test), fc.predict(split.test, weights="last")
    assert best.notna().all() and last.notna().all()
    importance = fc.feature_importance()
    assert "INSOLATION" in importance.index and "YEAR" in fc.feature_names_
    fc.save(tmp_path / "xgb")
    pd.testing.assert_series_equal(Forecaster.load(tmp_path / "xgb").predict(split.test), best)


def test_xgboost_matches_notebook_code(solar_frame):
    xgb = pytest.importorskip("xgboost")
    from energyprime.models import XGBoostRegressor

    features = ["INSOLATION", "T_AIR", "T_PANELS"]
    X, y = solar_frame[features], solar_frame["POWER"]
    eval_h, val_h = 5 * 24, 3 * 24
    X_tr, X_val, X_eval = X.iloc[:-eval_h - val_h], X.iloc[-eval_h - val_h:-eval_h], X.iloc[-eval_h:]
    y_tr, y_val = y.iloc[:-eval_h - val_h], y.iloc[-eval_h - val_h:-eval_h]
    params = {"objective": "reg:squarederror", "eval_metric": "rmse", "learning_rate": 0.03, "max_depth": 6,
              "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0, "seed": 42, "nthread": -1}
    booster = xgb.train(params, xgb.DMatrix(X_tr, label=y_tr), num_boost_round=500,
                        evals=[(xgb.DMatrix(X_tr, label=y_tr), "train"), (xgb.DMatrix(X_val, label=y_val), "valid")],
                        early_stopping_rounds=30, verbose_eval=False)
    expected = booster.predict(xgb.DMatrix(X_eval), iteration_range=(0, booster.best_iteration + 1))

    split = split_by_time(solar_frame, test=eval_h, val=val_h)
    fc = Forecaster(XGBoostRegressor(n_estimators=500, early_stopping_rounds=30), target="POWER",
                    features=features).fit(split.train, validation=split.val)
    np.testing.assert_allclose(fc.predict(split.test).to_numpy(), expected, rtol=1e-6)


@pytest.mark.slow
def test_lstm_windows_history_and_saving(split, tmp_path):
    pytest.importorskip("keras")
    fc = _fit_quietly(Forecaster.from_recipe(
        "lstm", lookback=6, model_params={"epochs": 3, "units": 8, "patience": None, "seed": 0}),
        split.train, validation=split.val)
    context = pd.concat([split.train.data, split.val.data])
    pred = fc.predict(split.test, history=context)
    assert pred.notna().all()
    assert fc.predict(split.test).iloc[:5].isna().all()      # no context -> first windows incomplete
    assert fc.history_.last_step == 3
    last = fc.predict(split.test, history=context, weights="last")
    assert last.notna().all()
    fc.save(tmp_path / "lstm")
    loaded = Forecaster.load(tmp_path / "lstm")
    np.testing.assert_allclose(loaded.predict(split.test, history=context), pred, rtol=1e-5)


@pytest.mark.slow
def test_svgp_interval_saving_and_resume(split, tmp_path):
    pytest.importorskip("gpytorch")
    from energyprime.models import SVGPRegressor

    params = {"epochs": 2, "mlp_hidden": (16,), "n_inducing": 16, "batch_size": 128, "seed": 0, "verbose": 0}
    fc = _fit_quietly(Forecaster.from_recipe("svgp", model_params=params), split.train, validation=split.val)
    interval = fc.predict_interval(split.test)
    assert list(interval.columns) == ["pred", "lower", "upper", "std"]
    assert (interval["lower"] <= interval["pred"] + 1e-9).all() and (interval["pred"] <= interval["upper"] + 1e-9).all()
    assert interval["pred"].between(split.train.y.min() - 1e-3, split.train.y.max() + 1e-3).all()
    fc.save(tmp_path / "svgp")
    loaded = Forecaster.load(tmp_path / "svgp")
    np.testing.assert_allclose(loaded.predict(split.test), interval["pred"], rtol=1e-6)

    X = np.random.default_rng(0).normal(size=(200, 3))
    y = X @ np.array([1.0, -2.0, 0.5])
    ckpt = tmp_path / "ckpt"
    SVGPRegressor(**{**params, "checkpoint_dir": str(ckpt)}).fit(X, y)
    resumed = SVGPRegressor(**{**params, "epochs": 4, "checkpoint_dir": str(ckpt), "resume": True}).fit(X, y)
    assert resumed.history_.last_step == 4


def test_checks_shuffled_target_and_permutation_importance(split):
    pytest.importorskip("sklearn")
    fc = _fit_quietly(Forecaster.from_recipe("linear"), split.train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shuffled = shuffled_target_check(fc, split.train, split.test)
    assert shuffled["r2_shuffled"] < 0.3
    importance = permutation_importance(fc, split.test, n_repeats=2)
    assert importance.iloc[0]["feature"] in {"INSOLATION", "T_PANELS"}


def test_model_kind_checks(split):
    pytest.importorskip("sklearn")
    with pytest.raises(ValueError, match="sequence models"):
        Forecaster.from_recipe("linear", lookback=24).fit(split.train)
