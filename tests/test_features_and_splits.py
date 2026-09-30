import numpy as np
import pandas as pd
import pytest

from energyprime import split_by_time
from energyprime.features import CalendarFeatures, CyclicEncoder, LagFeatures, RollingFeatures, make_windows
from energyprime.splits import expanding_window_splits


def test_cyclic_encoder_matches_notebook_formulas(solar_frame):
    out = CyclicEncoder({"hour": 24, "day_of_year": 365.25, "DIRECTION": 360}).fit_transform(solar_frame)
    assert "DIRECTION" not in out.columns
    hours = solar_frame.index.hour
    assert np.allclose(out["HOUR_SIN"], np.sin(2 * np.pi * hours / 24))
    assert np.allclose(out["DAY_OF_YEAR_COS"], np.cos(2 * np.pi * solar_frame.index.dayofyear / 365.25))
    assert np.allclose(out["DIRECTION_SIN"], np.sin(2 * np.pi * solar_frame["DIRECTION"] / 360))


def test_calendar_features(solar_frame):
    out = CalendarFeatures(["year", "weekday", "weekend"]).fit_transform(solar_frame)
    assert (out["YEAR"] == 2021).all()
    assert set(out["WEEKEND"]) == {0, 1}


def test_lags_are_time_aware():
    index = pd.date_range("2021-01-01", periods=6, freq="h").delete(3)  # 03:00 missing
    frame = pd.DataFrame({"P": [0.0, 1.0, 2.0, 4.0, 5.0]}, index=index)
    out = LagFeatures(["P"], lags=[1, 2], freq="1h").fit_transform(frame)
    assert np.isnan(out["P_lag_1"].iloc[3])            # 04:00 <- 03:00 is missing
    assert out["P_lag_2"].iloc[3] == 2.0                 # 04:00 <- 02:00
    assert out["P_lag_1"].iloc[4] == 4.0


def test_rolling_excludes_current_value():
    index = pd.date_range("2021-01-01", periods=5, freq="h")
    frame = pd.DataFrame({"P": [1.0, 2.0, 3.0, 4.0, 100.0]}, index=index)
    out = RollingFeatures(["P"], windows=[2], stats=["mean"]).fit_transform(frame)
    assert np.isnan(out["P_roll_mean_2"].iloc[1])
    assert out["P_roll_mean_2"].iloc[4] == 3.5


def notebook_windows(F, T, L):
    X, y, pos = [], [], []
    for t in range(L - 1, len(T)):
        X.append(F[t - L + 1: t + 1])
        y.append(T[t])
        pos.append(t)
    return np.array(X), np.array(y), np.array(pos)


def test_windows_match_notebook_and_skip_gaps():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(50, 3))
    X_ref, _, pos_ref = notebook_windows(values, np.zeros(50), 6)
    X, pos = make_windows(values, 6)
    assert np.array_equal(pos, pos_ref) and np.allclose(X, X_ref)

    index = pd.date_range("2021-01-01", periods=51, freq="h").delete(20)
    X_gap, pos_gap = make_windows(values, 6, index=index, freq="1h")
    assert not any(20 <= p <= 24 for p in pos_gap)   # windows crossing the gap are skipped
    assert len(pos_gap) == len(pos_ref) - 5
    values[30, 1] = np.nan
    _, pos_nan = make_windows(values, 6, index=index, freq="1h")
    assert not any(30 <= p <= 35 for p in pos_nan)


def test_split_rows_time_and_dates(solar_frame):
    by_rows = split_by_time(solar_frame, test=720 // 3, val=120)
    by_time = split_by_time(solar_frame, test="10D", val="5D")
    assert len(by_rows.test) == len(by_time.test) == 240
    assert len(by_rows.val) == len(by_time.val) == 120
    assert by_rows.train.index.max() < by_rows.val.index.min() < by_rows.test.index.min()
    by_date = split_by_time(solar_frame, test="2021-04-01")
    assert by_date.test.index.min() == pd.Timestamp("2021-04-01") and by_date.val is None
    with pytest.raises(ValueError):
        split_by_time(solar_frame, test=10_000)


def test_split_map_and_expanding_folds(solar_dataset):
    split = split_by_time(solar_dataset, test="5D", val="2D")
    assert split.test.spec is solar_dataset.spec
    doubled = split.map(lambda d: d.with_data(d.data * 2))
    assert doubled.train.data["POWER"].iloc[0] == 2 * split.train.data["POWER"].iloc[0]
    folds = expanding_window_splits(solar_dataset, n_splits=3, test="5D")
    assert len(folds) == 3
    ends = [f.test.data.index.max() for f in folds]
    assert ends == sorted(ends) and ends[-1] == solar_dataset.data.index.max()
