import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from energyprime import Dataset, StationSpec


def make_solar_frame(n_days: int = 40, seed: int = 0, start: str = "2021-03-01") -> pd.DataFrame:
    """Hourly solar-like data: power follows insolation and temperature, night power is small noise."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_days * 24, freq="h", name="time")
    hour = index.hour.to_numpy()
    sun = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None)
    clouds = rng.uniform(0.5, 1.0, len(index))
    insolation = 900 * sun * clouds
    t_air = 8 + 10 * sun + rng.normal(0, 1, len(index))
    t_panels = t_air + 15 * sun * clouds
    power = 50 * insolation / 1000 * (1 - 0.004 * (t_panels - 25)) + np.abs(rng.normal(0, 0.05, len(index)))
    direction = rng.uniform(0, 360, len(index))
    return pd.DataFrame({"POWER": power, "INSOLATION": insolation, "T_AIR": t_air,
                         "T_PANELS": t_panels, "DIRECTION": direction}, index=index)


SPEC = {
    "name": "synthetic",
    "capacity": 50,
    "time": {"freq": "1h"},
    "columns": {
        "POWER": {"role": "target", "unit": "MW", "limits": [-50, None]},
        "INSOLATION": {"role": "feature", "limits": [-50, None]},
        "T_AIR": {"role": "feature", "limits": [-50, 60]},
        "T_PANELS": {"role": "feature", "available": "history"},
        "DIRECTION": {"role": "feature", "cyclic": 360},
    },
}


@pytest.fixture
def solar_frame() -> pd.DataFrame:
    return make_solar_frame()


@pytest.fixture
def solar_spec() -> StationSpec:
    return StationSpec.from_dict(SPEC)


@pytest.fixture
def solar_dataset(solar_frame, solar_spec) -> Dataset:
    return Dataset(solar_frame, solar_spec)
