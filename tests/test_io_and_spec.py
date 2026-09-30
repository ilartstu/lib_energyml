import warnings

import numpy as np
import pandas as pd
import pytest

from energyprime import StationSpec, load_station, load_table, merge_by_time
from energyprime.datasets import raw_preview


def test_csv_with_metadata_semicolons_and_decimal_commas(tmp_path):
    text = (
        "Станция;SPP_2;;\n"
        "координаты;43.5;40.0;\n"
        "\n"
        "TIMESTAMP;POWER;T_AIR;COMMENT\n"
        "01.01.21 00:00;0,5;-1,25;ok\n"
        "01.01.21 01:00;1,5;-0,75;ok\n"
        "01.01.21 02:00;bad;0,5;ok\n"
        "01.01.21 03:00;2,0;1,0;ok\n"
    )
    path = tmp_path / "station.csv"
    path.write_bytes(text.encode("cp1251"))
    assert raw_preview(path)["suggested_header"] == 3
    ds = load_table(path, target="POWER", time_column="TIMESTAMP", time_format="%d.%m.%y %H:%M")
    assert list(ds.data.columns) == ["POWER", "T_AIR", "COMMENT"]
    assert ds.data["POWER"].tolist()[:2] == [0.5, 1.5]
    assert np.isnan(ds.data["POWER"].iloc[2])
    assert ds.info["values_not_numeric"] == {"POWER": 1}
    assert ds.target == "POWER" and ds.features == ["T_AIR"]
    assert ds.data.index[1] == pd.Timestamp("2021-01-01 01:00")


def test_excel_and_unsorted_input(tmp_path):
    frame = pd.DataFrame({"time": pd.to_datetime(["2021-01-01 02:00", "2021-01-01 00:00", "2021-01-01 01:00"]),
                          "POWER": [3.0, 1.0, 2.0]})
    path = tmp_path / "data.xlsx"
    frame.to_excel(path, index=False)
    ds = load_table(path, target="POWER")
    assert ds.info["was_sorted"] is False
    assert ds.data["POWER"].tolist() == [1.0, 2.0, 3.0]


def test_spec_roundtrip_and_validation(tmp_path):
    spec = StationSpec.from_dict({
        "name": "s", "capacity": 50, "time": {"column": "TIMESTAMP", "freq": "1h"},
        "columns": {"POWER": {"role": "target", "limits": [-50, None]}, "T_PANELS": {"available": "history"},
                    "DIRECTION": {"cyclic": 360}, "OTHER": "ignore"},
        "cleaning": {"missing": {"defaults": {"zeros": True}}},
    })
    assert spec.target == "POWER"
    assert spec.features == ["T_PANELS", "DIRECTION"]
    assert spec.availability("T_PANELS") == "history"
    assert spec.limits == {"POWER": (-50.0, None)}
    path = tmp_path / "spec.yaml"
    spec.to_yaml(path)
    assert StationSpec.from_yaml(path).to_dict() == spec.to_dict()
    with pytest.raises(ValueError, match="avaliable"):
        StationSpec.from_dict({"columns": {"A": {"avaliable": "forecast"}}})
    with pytest.raises(ValueError, match="several targets"):
        StationSpec.from_dict({"columns": {"A": "target", "B": "target"}})


def test_load_station_roles_and_warnings(tmp_path, solar_frame):
    path = tmp_path / "solar.csv"
    solar_frame.assign(EXTRA=1.0).to_csv(path)
    spec = {"name": "x", "time": {"column": "time"},
            "columns": {"POWER": "target", "T_AIR": "feature", "MISSING_COL": "feature"}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = load_station(path, spec)
    assert any("MISSING_COL" in str(w.message) for w in caught)
    assert ds.features == ["T_AIR"]
    assert "EXTRA" in ds.info["ignored_columns"]
    assert ds.reference_power() == (pytest.approx(ds.y.max()), "max")


def test_merge_by_time_prefixes_only_collisions(solar_frame):
    weather = solar_frame[["T_AIR"]].iloc[::2] + 1
    merged = merge_by_time({"station": solar_frame[["POWER", "T_AIR"]], "nwp": weather})
    assert {"station_T_AIR", "nwp_T_AIR", "POWER"} <= set(merged.columns)
    assert merged["nwp_T_AIR"].isna().sum() == len(solar_frame) // 2


def test_example_specs_are_valid():
    from pathlib import Path

    from energyprime import Cleaner

    specs = sorted((Path(__file__).parent.parent / "examples" / "specs").glob("*.yaml"))
    assert len(specs) == 3
    rng = np.random.default_rng(0)
    for path in specs:
        spec = StationSpec.from_yaml(path)
        assert spec.target == "POWER"
        index = pd.date_range("2021-01-01", periods=200, freq="h")
        frame = pd.DataFrame(rng.uniform(1, 10, (200, len(spec.columns))), index=index, columns=list(spec.columns))
        plan = Cleaner.from_spec(spec).plan(frame)
        assert plan["rows_to_drop"] == 0
        Cleaner.from_spec(spec).fit_transform(frame)
