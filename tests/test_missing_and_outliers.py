import numpy as np
import pandas as pd
import pytest

from energyprime.preprocessing import (
    DeleteLargeGaps,
    ForwardFill,
    LinearInterpolation,
    MedianImputer,
    MissingValueProcessor,
    NanToZero,
    OutlierProcessor,
)
from energyprime.preprocessing.missing import ColumnRules, parse_runs


def _series_frame(values, extra=None):
    index = pd.date_range("2021-01-01", periods=len(values), freq="h")
    data = {"A": values}
    if extra is not None:
        data["B"] = extra
    return pd.DataFrame(data, index=index)


def test_parse_runs_both_forms_and_overlap():
    rules = parse_runs({"keep": [1, 3], "interpolate": [4, 24], "drop_rows": [25, None]})
    assert [(r.min_length, r.max_length, r.action) for r in rules] == [
        (1, 3, "keep"), (4, 24, "interpolate"), (25, None, "drop_rows")]
    rules2 = parse_runs([{"length": [1, 2], "action": "ffill"}])
    assert rules2[0].action == "ffill"
    with pytest.raises(ValueError, match="overlap"):
        parse_runs({"keep": [1, 5], "interpolate": [4, 24]})
    with pytest.raises(ValueError, match="Unknown action"):
        parse_runs({"explode": [1, 2]})
    with pytest.raises(ValueError, match="Unknown missing-value option"):
        ColumnRules.from_dict({"zeroes": True})


def test_keep_interpolate_drop_by_length():
    values = [1.0, np.nan, 3.0,                     # 1 NaN -> keep
              4.0, np.nan, np.nan, np.nan, np.nan, 9.0,  # 4 NaN -> interpolate
              10.0] + [np.nan] * 5 + [16.0]          # 5 NaN -> drop rows
    df = _series_frame(values)
    proc = MissingValueProcessor({"defaults": {"runs": {"keep": [1, 3], "interpolate": [4, 4],
                                                        "drop_rows": [5, None]}}})
    out = proc.fit_transform(df)
    assert len(out) == len(df) - 5
    assert np.isnan(out["A"].iloc[1])
    assert out["A"].iloc[4:8].tolist() == [5.0, 6.0, 7.0, 8.0]
    report = proc.get_report()
    assert report["rows_dropped"] == 5
    assert report["fills"].loc["A", "interpolated"] == 4
    assert report["removed_periods"]["rows"].tolist() == [5]


def test_zeros_merge_versus_separate():
    values = [5.0, 0.0, 0.0, np.nan, 0.0, 7.0]
    df = _series_frame(values)
    merged = MissingValueProcessor({"defaults": {"zeros": True, "merge": True,
                                                 "runs": {"interpolate": [4, 4]}}}).fit_transform(df)
    assert np.allclose(merged["A"].to_numpy(), [5.0, 5.4, 5.8, 6.2, 6.6, 7.0])
    separate = MissingValueProcessor({"defaults": {"zeros": True, "merge": False,
                                                   "runs": {"interpolate": [4, 4]}}}).fit_transform(df)
    assert separate["A"].iloc[1] == 0.0 and np.isnan(separate["A"].iloc[3])
    per_kind = MissingValueProcessor({"defaults": {"zeros": True, "merge": False, "runs": {},
                                                   "nan_runs": {"fill_zero": [1, 1]},
                                                   "zero_runs": {"interpolate": [2, 2]}}}).fit_transform(df)
    assert per_kind["A"].iloc[3] == 0.0
    assert per_kind["A"].iloc[1] != 0.0


def test_nan_to_zero_and_edges():
    df = _series_frame([np.nan, np.nan, 1.0, np.nan, 3.0, np.nan])
    out = NanToZero().fit_transform(df)
    assert out["A"].tolist() == [0.0, 0.0, 1.0, 0.0, 3.0, 0.0]
    proc = LinearInterpolation(max_gap=2)
    out = proc.fit_transform(df)
    assert out["A"].iloc[3] == 2.0
    assert np.isnan(out["A"].iloc[0]) and np.isnan(out["A"].iloc[5])
    assert proc.get_report()["fills"].loc["A", "interpolation_failed"] == 3


def test_drop_scope_only_used_columns():
    df = _series_frame([1.0] * 8, extra=[1.0, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, 1.0])
    proc = MissingValueProcessor({"defaults": {"runs": {"drop_rows": [3, None]}}}, drop_scope=["A"])
    assert len(proc.fit_transform(df)) == 8
    assert len(DeleteLargeGaps(min_gap=3).fit_transform(df)) == 3


def test_forward_fill_and_median_imputer():
    df = _series_frame([1.0, np.nan, np.nan, np.nan, 5.0])
    out = ForwardFill(max_gap=2).fit_transform(df)
    assert np.isnan(out["A"].iloc[1])  # a run of 3 is longer than max_gap
    out = ForwardFill(max_gap=3).fit_transform(df)
    assert out["A"].tolist() == [1.0, 1.0, 1.0, 1.0, 5.0]

    index = pd.date_range("2021-01-01", periods=48, freq="h")
    frame = pd.DataFrame({"A": np.tile(np.arange(24, dtype=float), 2)}, index=index)
    frame.iloc[30, 0] = np.nan
    imputer = MedianImputer(columns=["A"], groupby=["hour"]).fit(frame)
    assert imputer.transform(frame)["A"].iloc[30] == 6.0


def test_outlier_detectors_and_actions():
    rng = np.random.default_rng(1)
    values = rng.normal(10, 1, 200)
    values[50] = 100.0
    values[60] = 0.0      # zero: missing, not an outlier
    values[70] = -80.0    # below physical limit
    df = _series_frame(values)
    proc = OutlierProcessor(detectors={"zscore": {"threshold": 4}, "iqr": {"factor": 3}}, apply=["zscore"],
                            limits={"A": (-50, None)}, ignore_zeros=True, action="set_nan")
    out = proc.fit_transform(df)
    assert np.isnan(out["A"].iloc[50]) and np.isnan(out["A"].iloc[70])
    assert out["A"].iloc[60] == 0.0
    summary = proc.get_report()["summary"]
    assert summary.loc["A", "physical"] == 1 and summary.loc["A", "applied"] == 2
    assert {"zscore", "iqr", "physical", "applied"} <= set(proc.diagnose(df).columns)

    clipped = OutlierProcessor(detectors={"zscore": {"threshold": 4}}, action="clip").fit_transform(df)
    mean, std = np.mean(values[values != 0]), np.std(values[values != 0], ddof=1)
    assert clipped["A"].iloc[50] < 100.0 and clipped["A"].iloc[50] <= mean + 4 * std + 1e-9

    flagged = OutlierProcessor(detectors={"zscore": {"threshold": 4}}, action="flag").fit_transform(df)
    assert flagged["A_outlier"].sum() >= 1

    dropped = OutlierProcessor(detectors={"zscore": {"threshold": 4}}, action="drop_rows").fit_transform(df)
    assert len(dropped) < len(df)


def test_outlier_statistics_come_from_fit_data():
    train = _series_frame(np.r_[np.full(50, 10.0), np.full(50, 12.0)])
    test = _series_frame([11.0, 30.0])
    proc = OutlierProcessor(detectors={"zscore": {"threshold": 3}}, action="set_nan").fit(train)
    out = proc.transform(test)
    assert out["A"].iloc[0] == 11.0 and np.isnan(out["A"].iloc[1])
