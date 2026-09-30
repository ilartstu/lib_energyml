"""Sanity checks that work with any Forecaster."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.validation import as_frame
from .metrics import evaluate_regression


def shuffled_target_check(forecaster, train, test, val=None, history=None, seed: int = 42,
                          metric: str = "r2") -> dict[str, float]:
    """Train an unfitted copy on a shuffled target; the score should collapse.

    R² close to 0 (or negative) means the model learns from the features and
    not from some leak. This is STEP 9 of the XGBoost notebook for any model.
    """
    copy = forecaster.clone()
    train_frame = as_frame(train).copy()
    target = copy.resolve_target(train)
    rng = np.random.default_rng(seed)
    train_frame[target] = rng.permutation(train_frame[target].to_numpy())
    shuffled_train = train.with_data(train_frame) if hasattr(train, "with_data") else train_frame
    copy.fit(shuffled_train, validation=val)
    pred = copy.predict(test, history=history)
    y_true = as_frame(test)[target]
    return {f"{metric}_shuffled": evaluate_regression(y_true, pred.reindex(y_true.index), [metric])[metric]}


def permutation_importance(forecaster, data, history=None, columns=None, metric: str = "rmse",
                           n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """How much the error grows when one input column is shuffled.

    Works for any model, including LSTM (the column is shuffled before the
    windows are built). Positive importance = the column helps.
    """
    frame = as_frame(data)
    target = forecaster.target_
    y_true = frame[target]
    base_pred = forecaster.predict(data, history=history)
    base = evaluate_regression(y_true, base_pred.reindex(y_true.index), [metric])[metric]
    rng = np.random.default_rng(seed)
    cols = columns or [c for c in forecaster.input_columns_ if c != target]
    rows = []
    for col in cols:
        scores = []
        for _ in range(n_repeats):
            shuffled = frame.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            part = data.with_data(shuffled) if hasattr(data, "with_data") else shuffled
            pred = forecaster.predict(part, history=history)
            scores.append(evaluate_regression(y_true, pred.reindex(y_true.index), [metric])[metric])
        sign = -1.0 if metric == "r2" else 1.0
        diffs = sign * (np.asarray(scores) - base)
        rows.append({"feature": col, "importance": diffs.mean(), "std": diffs.std()})
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
