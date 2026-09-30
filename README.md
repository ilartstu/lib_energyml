# EnergyPrime

Python library for solar and wind power forecasting from weather data:
data loading and cleaning, feature engineering, models (XGBoost, LSTM, SVGP),
evaluation and visualization.

Status: early development (0.1), the API may still change.

## Installation

Inside your environment (for example the conda env with PyTorch / TensorFlow):

```bash
pip install -e ".[all]"
```

`-e` (editable) means changes in `src/` are visible immediately, e.g. in Jupyter with `%autoreload`.
Extras install only what a model needs: `.[xgboost]`, `.[lstm]` (Keras + TensorFlow),
`.[gp]` (PyTorch + GPyTorch), `.[sklearn]`, `.[dev]` (pytest). `import energyprime` itself never
imports XGBoost, TensorFlow or PyTorch — they are loaded with the model that needs them.

## Quick start

```python
import pandas as pd
from energyprime import Cleaner, Forecaster, load_station, split_by_time
from energyprime import evaluation, visualization as viz

ds = load_station("solar_dataset_full_1.csv", "examples/specs/china_site1_hourly.yaml")

cleaner = Cleaner.from_spec(ds.spec)
print(cleaner.plan(ds)["rows_to_drop"])            # dry run: what the rules would do

split = split_by_time(ds, test="30D", val="5D")      # or test=720, val=120 (rows)
cleaner.fit(split.train)                             # outlier statistics from train only
split = cleaner.transform(split)

fc = Forecaster.from_recipe("xgboost")               # or "lstm", "svgp", "linear", "random_forest"
fc.fit(split.train, validation=split.val)

history = pd.concat([split.train.data, split.val.data])
pred = fc.predict(split.test, history=history)
fc.evaluate(split.test, history=history, percent=True)   # nMAE, nRMSE (% of capacity), R²
daily = evaluation.evaluate_by_window(split.test.y, pred, window=24, norm=ds.capacity)
viz.plot_forecast(split.test.y[-120:], pred[-120:])

fc.save("runs/xgb_site1")                            # later: Forecaster.load("runs/xgb_site1")
```

A longer walk-through is in [`examples/quickstart.ipynb`](examples/quickstart.ipynb).

## Station specification

One YAML file per station describes the columns and the cleaning rules
(examples in [`examples/specs`](examples/specs)):

```yaml
name: SPP_2
capacity: 50                 # MW; if null, errors are normalised by the maximum instead
time: {column: TIMESTAMP, format: "%d.%m.%y %H:%M", freq: 1h}
columns:
  POWER:     {role: target, unit: MW, limits: [-50, null]}
  T_AIR:     {role: feature, limits: [-50, 60]}
  T_PANELS:  {role: feature, available: history}   # measured only, unknown at a real forecast
  DIRECTION: {role: feature, cyclic: 360}           # encoded as sin/cos automatically
  OTHER:     ignore                                 # columns not listed are ignored too
cleaning: {...}
```

* `role` — `target`, `feature` or `ignore`.
* `available` — `forecast` (e.g. from weather forecasts), `history` (sensors only) or `always`.
  Training with a `history`-only feature gives a warning: the score assumes it is known.
* `limits` — physical range; values outside are outliers.

## Cleaning rules

Per column: what counts as missing, and what to do with a run of missing values depending on its
length (both bounds inclusive):

```yaml
cleaning:
  missing:
    defaults:
      nan: true
      zeros: false            # true: an exact 0 is missing (0.001 is not)
      merge: true             # "0 0 NaN 0" is one run of 4
      nan_to_zero: false
      runs: {keep: [1, 3], interpolate: [4, 24], drop_rows: [25, null]}
    columns:
      POWER: {zeros: true}
  outliers:
    detectors: {zscore: {threshold: 5}, iqr: {factor: 4}, hampel: {window: 72, n_sigmas: 12}}
    apply: [zscore]           # the others only appear in the diagnostics
    action: interpolate       # set_nan, clip, drop_rows, flag
  exclude:
    - {start: "2020-06-01", end: "2020-06-03", columns: [POWER]}
```

Actions: `keep`, `interpolate`, `fill_zero`, `ffill`, `bfill`, `set_nan`, `drop_rows`; separate
rules for NaN and zero runs via `nan_runs` / `zero_runs` (with `merge: false`). Only target and
feature columns can remove rows. The steps always run in the same order: time axis → exclusions
→ `nan_to_zero` → rows in `drop_rows` runs removed → outliers detected → fills → report.

Diagnostics (they never change the data): `analysis.describe_columns` (NaN / zeros per column),
`analysis.run_length_table` (how many runs of each length), `analysis.find_blocks`,
`analysis.missing_map`, `Cleaner.plan`. After cleaning, `cleaner.get_report()` holds the removed
periods, outlier counts per detector, the outlier mask and the fills per column.

## Recipes and the Forecaster

`Forecaster` = feature engineering + scaling + target transform + model; everything that must be
identical in training and in a real forecast. A recipe is a model with the preparation that suits
it (see `energyprime.models.list_recipes()`):

| recipe | model | features | scaling | target |
|---|---|---|---|---|
| `xgboost` | XGBoost (early stopping) | weather + YEAR + cyclic hour / day of year | — | as is |
| `lstm` | Keras LSTM(100) on 24 h windows | weather + cyclic time | min-max | min-max |
| `svgp` | GPyTorch SVGP, MLP mean, RBF-ARD | weather + YEAR + cyclic time | max-abs | min-max + logit |
| `linear`, `random_forest` | scikit-learn baselines | weather + cyclic time | standard / — | as is |

Everything is overridable: `Forecaster.from_recipe("lstm", lookback=48, model_params={"epochs": 50})`,
`Forecaster.from_recipe("xgboost", lags={"POWER": [1, 24]})`, or build one directly:
`Forecaster(XGBoostRegressor(), features=[...], feature_scaling="minmax")`.

* **Weights.** Models keep the best-epoch and the last-epoch weights:
  `predict(X, weights="best" | "last")`.
* **History.** `predict(data, history=...)` gives lags and LSTM windows the data before `data`.
  With lags of the target, `predict_recursive` feeds the model its own predictions (a real
  multi-step forecast).
* **Uncertainty.** `predict_interval` (SVGP) returns `pred`, `lower`, `upper` = mean ± k·std mapped
  back to MW.
* **Checks.** `evaluation.shuffled_target_check` (R² on a shuffled target should be ≈ 0) and
  `evaluation.permutation_importance` work with any model.

## Package layout

```
src/energyprime/
├── core/            base classes, Pipeline, registries, station spec, training history
├── datasets/        Dataset, reading CSV/Excel, merging sources by time
├── analysis/        NaN/zero statistics, run lengths, blocks, correlation, ACF, spectrum
├── preprocessing/   Cleaner, time axis, missing-value rules, outliers, scaling, target transforms
├── features/        calendar, cyclic (sin/cos), lags, rolling statistics, windows
├── splits.py        split by time, expanding-window folds
├── models/          XGBoost, LSTM (Keras), SVGP (GPyTorch), scikit-learn, recipes
├── training/        PyTorch epoch loop: best/last weights, checkpoints, explicit resume
├── forecaster.py    Forecaster
├── evaluation/      metrics, metrics per window / group, CRPS, sanity checks
└── visualization/   plots of data and forecasts (return a Figure, never show/save)
```

The library never prints results, shows or saves figures, changes global settings on import or
silently picks up files left from a previous run.

## Checked against the original notebooks

On `solar_dataset_full_1.csv` and the raw Excel of site 1:

* cleaning (SVGP notebook, STEP 3) — identical rows and values;
* XGBoost — identical predictions, best iteration and window metrics;
* LSTM and SVGP (fixed seeds, 5 epochs) — identical losses per epoch and predictions.

The unit tests (`tests/`) include the notebook's cleaning code as a reference.

## Development

```bash
pip install -e ".[all,dev]"
pytest                    # everything, including short LSTM/SVGP trainings
pytest -m "not slow"      # without the neural models
```

## Roadmap

* experiment framework on top of the library: `Experiment`, `Benchmark`, YAML configs, CLI, run folders;
* several series (wind turbines): `core.PerSeries` already applies a step per turbine; spatial and
  wake features will follow;
* GUI: loads a saved `Forecaster` and calls `predict` on weather forecasts.
