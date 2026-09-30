import numpy as np
import pandas as pd
import pytest

from energyprime import Cleaner, StationSpec, split_by_time
from energyprime.preprocessing import TimeProcessor

from .notebook_reference import svgp_notebook_clean

SVGP_CONFIG = {
    "time": {"duplicates": "keep"},
    "missing": {"defaults": {"nan": True, "zeros": True, "merge": True,
                             "runs": {"keep": [1, 3], "interpolate": [4, 24], "drop_rows": [25, None]}}},
    "outliers": {"detectors": {"zscore": {"threshold": 5.0}, "iqr": {"factor": 4.0},
                               "hampel": {"window": 72, "n_sigmas": 12.0}},
                 "apply": ["zscore"], "action": "interpolate"},
}
LIMITS = {"POWER": (-50, None), "T_AIR": (-50, 60), "DIRECTION": (None, None)}


def dirty_frame(seed=3):
    rng = np.random.default_rng(seed)
    n = 24 * 60
    index = pd.date_range("2021-01-01", periods=n, freq="h", name="time")
    frame = pd.DataFrame({
        "POWER": rng.uniform(1, 40, n),
        "T_AIR": rng.normal(10, 5, n),
        "DIRECTION": rng.uniform(1, 359, n),
    }, index=index)
    frame.iloc[100:103, 0] = 0.0          # short zeros -> keep
    frame.iloc[200:210, 0] = np.nan       # 10 NaN -> interpolate
    frame.iloc[205:209, 1] = 0.0
    frame.iloc[400:440, 1] = np.nan       # 40 -> drop rows
    frame.iloc[700:702, 2] = np.nan
    frame.iloc[702:705, 2] = 0.0          # merged NaN+zero run of 5 -> interpolate
    frame.iloc[900, 0] = 400.0            # z-score outlier
    frame.iloc[950, 1] = 75.0             # above physical limit
    frame.iloc[1000:1030, 0] = 0.0        # 30 zeros -> drop rows
    frame.iloc[0:5, 1] = np.nan           # edge run -> cannot interpolate -> stays NaN
    return frame


def test_cleaner_reproduces_svgp_notebook():
    frame = dirty_frame()
    columns = list(frame.columns)
    expected = svgp_notebook_clean(frame, columns, LIMITS)
    cleaner = Cleaner(SVGP_CONFIG, limits=LIMITS)
    result = cleaner.fit_transform(frame)
    assert result.index.equals(expected.index)
    np.testing.assert_allclose(result[columns].to_numpy(), expected[columns].to_numpy(), equal_nan=True)
    report = cleaner.get_report()
    assert report["rows_dropped"] == 40 + 30
    assert report["outliers"].loc["POWER", "zscore"] >= 1
    assert report["outliers"].loc["T_AIR", "physical"] == 1
    assert set(report["removed_periods"]["reason"]) == {"T_AIR", "POWER"}


def test_plan_is_a_dry_run():
    frame = dirty_frame()
    cleaner = Cleaner(SVGP_CONFIG, limits=LIMITS)
    plan = cleaner.plan(frame)
    assert plan["rows_to_drop"] == 70
    runs = plan["runs"]
    assert set(runs["action"]) == {"keep", "interpolate", "drop_rows"}
    assert frame.isna().sum().sum() > 0  # input untouched


def test_fit_on_train_transform_split():
    frame = dirty_frame()
    split = split_by_time(frame, test=24 * 10, val=24 * 5)
    cleaner = Cleaner(SVGP_CONFIG, limits=LIMITS)
    cleaned = cleaner.fit(split).transform(split)
    assert len(cleaned.test) == 240 and len(cleaned.val) == 120
    reports = cleaner.get_report()
    assert set(reports) == {"train", "val", "test"}
    assert reports["train"]["rows_dropped"] == 70 and reports["test"]["rows_dropped"] == 0
    stats = cleaner.outliers_.fitted_detectors_["POWER"]["zscore"]
    train_power = split.train["POWER"].drop(split.train.index[400:440])  # rows removed by T_AIR gap
    clean_train = train_power[(train_power != 0) & train_power.notna()]
    assert stats.mean_ == pytest.approx(clean_train.mean(), rel=1e-12)


def test_drop_scope_from_spec_and_exclusions():
    frame = dirty_frame()
    spec = StationSpec.from_dict({
        "columns": {"POWER": {"role": "target", "limits": [-50, None]}, "DIRECTION": "feature",
                    "T_AIR": "ignore"},
        "cleaning": {**SVGP_CONFIG,
                     "exclude": [{"start": "2021-01-20", "end": "2021-01-20 05:00", "columns": ["POWER"]}]},
    })
    cleaner = Cleaner.from_spec(spec)
    out = cleaner.fit_transform(frame)
    assert cleaner.get_report()["rows_dropped"] == 30          # T_AIR (ignored) no longer drops rows
    assert cleaner.get_report()["excluded_values"] == 6
    assert out.loc["2021-01-20 00:00":"2021-01-20 05:00", "POWER"].notna().all()  # 6 NaN -> interpolated


def test_time_processor_duplicates_and_grid():
    index = pd.to_datetime(["2021-01-01 00:00", "2021-01-01 01:00", "2021-01-01 01:00", "2021-01-01 03:00"])
    frame = pd.DataFrame({"A": [1.0, 2.0, 4.0, 5.0]}, index=index)
    proc = TimeProcessor(duplicates="mean", freq="1h", fill_missing_timestamps=True)
    out = proc.fit_transform(frame)
    assert out["A"].tolist()[:2] == [1.0, 3.0]
    assert np.isnan(out["A"].iloc[2]) and len(out) == 4
    assert proc.get_report()["duplicate_timestamps"] == 1
    assert proc.get_report()["inserted_rows"] == 1


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown cleaning option"):
        Cleaner({"mising": {}}).fit(dirty_frame())
