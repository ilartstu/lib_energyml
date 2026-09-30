"""EnergyPrime — solar and wind power forecasting from weather data.

Main entry points::

    from energyprime import load_station, split_by_time, Cleaner, Forecaster

Subpackages: ``datasets``, ``analysis``, ``preprocessing``, ``features``,
``models``, ``evaluation``, ``visualization``. Importing ``energyprime`` does
not import XGBoost, TensorFlow/Keras or PyTorch — they load with the model
that needs them.
"""
from .core.schema import ColumnSpec, StationSpec
from .datasets import Dataset, load_station, load_table, merge_by_time
from .forecaster import Forecaster
from .preprocessing import Cleaner
from .splits import Split, expanding_window_splits, split_by_time

__version__ = "0.1.0"

__all__ = [
    "__version__", "StationSpec", "ColumnSpec", "Dataset", "load_station", "load_table", "merge_by_time",
    "Cleaner", "Forecaster", "Split", "split_by_time", "expanding_window_splits",
]
