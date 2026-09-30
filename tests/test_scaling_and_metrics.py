import numpy as np
import pandas as pd
import pytest

from energyprime.evaluation import (
    crps_gaussian,
    evaluate_by_group,
    evaluate_by_window,
    evaluate_regression,
    mae,
    r2,
    rmse,
    summarize_windows,
)
from energyprime.preprocessing import MinMaxLogitTarget, MinMaxTarget, ScalingProcessor


def test_minmax_logit_matches_svgp_notebook():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 50, 500)
    transform = MinMaxLogitTarget().fit(y[:400])
    z = transform.transform(y)
    y_min, y_max = y[:400].min(), y[:400].max()
    ref = (y - y_min) / (y_max - y_min + 1e-6)
    ref = np.clip(ref, 1e-6, 1 - 1e-6)
    ref = np.clip(np.log(ref / (1 - ref)), -10, 10)
    assert np.allclose(z, ref)
    inside = (y >= y_min) & (y <= y_max)
    # the logit is clipped at ±10, so values within scale * expit(-10) of min/max are not exact
    tolerance = (y_max - y_min) * 1 / (1 + np.exp(10)) + 1e-6
    assert np.allclose(transform.inverse_transform(z)[inside], y[inside], atol=tolerance)


def test_minmax_target_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.preprocessing")
    y = np.array([3.0, 5.0, 11.0, 7.0])
    ours = MinMaxTarget().fit(y[:3]).transform(y)
    ref = sklearn.MinMaxScaler().fit(y[:3, None]).transform(y[:, None]).ravel()
    assert np.allclose(ours, ref)


@pytest.mark.parametrize("method", ["standard", "minmax", "robust", "maxabs"])
def test_scaling_roundtrip(method, solar_frame):
    scaler = ScalingProcessor(method, columns=["T_AIR", "INSOLATION"])
    scaled = scaler.fit_transform(solar_frame)
    back = scaler.inverse_transform(scaled)
    assert np.allclose(back[["T_AIR", "INSOLATION"]], solar_frame[["T_AIR", "INSOLATION"]])
    assert scaled["POWER"].equals(solar_frame["POWER"])


def test_maxabs_with_capacity():
    frame = pd.DataFrame({"POWER": [0.0, 25.0, 40.0]}, index=pd.date_range("2021", periods=3, freq="h"))
    out = ScalingProcessor("maxabs", scale={"POWER": 50}).fit_transform(frame)
    assert out["POWER"].tolist() == [0.0, 0.5, 0.8]
    with pytest.raises(ValueError):
        ScalingProcessor("minmax", scale={"POWER": 50}).fit(frame)


def test_metrics_match_sklearn():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y, p = rng.normal(size=100), rng.normal(size=100)
    assert mae(y, p) == pytest.approx(metrics.mean_absolute_error(y, p))
    assert rmse(y, p) == pytest.approx(np.sqrt(metrics.mean_squared_error(y, p)))
    assert r2(y, p) == pytest.approx(metrics.r2_score(y, p))
    assert r2(np.ones(5), np.ones(5) * 2) == metrics.r2_score(np.ones(5), np.ones(5) * 2)


def test_crps_matches_properscoring():
    ps = pytest.importorskip("properscoring")
    rng = np.random.default_rng(2)
    y, mu, sigma = rng.normal(size=50), rng.normal(size=50), rng.uniform(0.1, 2, 50)
    assert crps_gaussian(y, mu, sigma) == pytest.approx(np.mean(ps.crps_gaussian(y, mu, sigma)))


def notebook_rolling_metrics(y_true, y_pred, horizon, step, norm):
    """The notebooks' rolling_origin_metrics without printing."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    rows = []
    for s in range(0, len(y_true) - horizon + 1, step):
        e = s + horizon
        rows.append({"MAE": mean_absolute_error(y_true[s:e], y_pred[s:e]) / norm,
                     "RMSE": np.sqrt(mean_squared_error(y_true[s:e], y_pred[s:e])) / norm,
                     "R2": r2_score(y_true[s:e], y_pred[s:e])})
    return pd.DataFrame(rows)


def test_window_metrics_match_notebook():
    pytest.importorskip("sklearn")
    index = pd.date_range("2021-01-01", periods=720, freq="h")
    rng = np.random.default_rng(3)
    y = pd.Series(rng.uniform(0, 50, 720), index=index)
    p = y + rng.normal(0, 3, 720)
    ref = notebook_rolling_metrics(y.to_numpy(), p.to_numpy(), 120, 24, 50.0)
    ours = evaluate_by_window(y, p, window=120, step=24, norm=50.0)
    assert len(ours) == len(ref) == 26
    assert np.allclose(ours["nmae"], ref["MAE"]) and np.allclose(ours["r2"], ref["R2"])
    by_time = evaluate_by_window(y, p, window="24h", norm=50.0)
    by_rows = evaluate_by_window(y, p, window=24, norm=50.0)
    assert np.allclose(by_time["nrmse"], by_rows["nrmse"])
    assert list(summarize_windows(ours).index) == ["mean", "std", "min", "max"]


def test_evaluate_regression_options():
    y = np.array([0.0, 10.0, 20.0])
    p = np.array([1.0, 9.0, 23.0])
    scores = evaluate_regression(y, p, ["nmae", "nrmse"], norm=50, percent=True)
    assert scores["nmae"] == pytest.approx(mae(y, p) / 50 * 100)
    with pytest.raises(ValueError, match="needs"):
        evaluate_regression(y, p, ["nmae"])
    groups = evaluate_by_group(pd.Series(y, index=pd.date_range("2021", periods=3, freq="h")), p,
                               by="hour", metrics=["mae"])
    assert list(groups.index) == [0, 1, 2]
