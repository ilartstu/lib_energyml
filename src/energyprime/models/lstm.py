"""LSTM (Keras) — the LSTM notebook as a model.

Default architecture = the notebook: ``LSTM(100) -> Dropout(0.2) -> Dense(1)``,
MSE loss, Adam, 100 epochs, batch 64, early stopping with patience 10,
no shuffling. Input: windows ``(samples, lookback, features)`` built by the
Forecaster (``lookback=24`` in the recipe).

Both the best-epoch and the last-epoch weights are kept; choose with
``predict(X, weights="best" | "last")``.
"""
from __future__ import annotations

import time
from pathlib import Path

import keras
import numpy as np

from ..core.base import BaseModel
from ..core.history import TrainingHistory


class _BestLastWeights(keras.callbacks.Callback):
    """Remembers the weights of the epoch with the lowest monitored loss."""

    def __init__(self, monitor: str):
        super().__init__()
        self.monitor = monitor
        self.best = np.inf
        self.best_epoch = None
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        value = (logs or {}).get(self.monitor)
        if value is not None and value < self.best:
            self.best, self.best_epoch = float(value), epoch + 1
            self.best_weights = self.model.get_weights()


class LSTMRegressor(BaseModel):
    """Keras LSTM regressor on windows.

    >>> LSTMRegressor(units=100, dropout=0.2, epochs=100, batch_size=64, patience=10, seed=42)
    """

    input_kind = "sequence"

    def __init__(self, units: int = 100, dropout: float = 0.2, layers: int = 1, epochs: int = 100,
                 batch_size: int = 64, patience: int | None = 10, learning_rate: float = 0.001,
                 loss: str = "mean_squared_error", shuffle: bool = False, seed: int | None = None,
                 verbose: int = 0):
        self.units = units
        self.dropout = dropout
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.loss = loss
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose

    def _build(self, lookback: int, n_features: int) -> keras.Model:
        stack = [keras.Input(shape=(lookback, n_features))]
        for i in range(self.layers):
            stack.append(keras.layers.LSTM(self.units, return_sequences=i < self.layers - 1))
            stack.append(keras.layers.Dropout(self.dropout))
        stack.append(keras.layers.Dense(1))
        model = keras.Sequential(stack)
        model.compile(loss=self.loss, optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def fit(self, X, y, X_val=None, y_val=None) -> "LSTMRegressor":
        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32").ravel()
        if X.ndim != 3:
            raise ValueError(f"LSTM expects windows (samples, lookback, features), got shape {X.shape}")
        if self.seed is not None:
            keras.utils.set_random_seed(self.seed)
        model = self._build(X.shape[1], X.shape[2])
        has_val = X_val is not None and y_val is not None and len(y_val) > 0
        validation = (np.asarray(X_val, dtype="float32"), np.asarray(y_val, dtype="float32").ravel()) \
            if has_val else None
        monitor = "val_loss" if has_val else "loss"
        tracker = _BestLastWeights(monitor)
        callbacks = [tracker]
        if self.patience:
            callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=self.patience,
                                                           restore_best_weights=False))
        t0 = time.perf_counter()
        result = model.fit(X, y, validation_data=validation, epochs=self.epochs, batch_size=self.batch_size,
                           shuffle=self.shuffle, callbacks=callbacks, verbose=self.verbose)
        train_time = time.perf_counter() - t0

        last = model.get_weights()
        self.model_ = model
        self.weights_ = {"best": tracker.best_weights if tracker.best_weights is not None else last,
                         "last": last}
        self.input_shape_ = (int(X.shape[1]), int(X.shape[2]))
        losses = result.history
        self.history_ = TrainingHistory(
            train=[float(v) for v in losses["loss"]], val=[float(v) for v in losses.get("val_loss", [])],
            metric="mse" if self.loss in ("mse", "mean_squared_error") else self.loss, step_name="epoch",
            best_step=tracker.best_epoch, train_time_sec=train_time,
            stopped_early=len(losses["loss"]) < self.epochs,
        )
        self._active_weights = "last"
        self.fitted_ = True
        self._use_weights("best")
        return self

    def _use_weights(self, weights: str) -> None:
        self._check_weights(weights)
        if getattr(self, "_active_weights", None) != weights:
            self.model_.set_weights(self.weights_[weights])
            self._active_weights = weights

    def predict(self, X, weights: str = "best") -> np.ndarray:
        self._check_fitted()
        self._use_weights(weights)
        X = np.asarray(X, dtype="float32")
        if len(X) == 0:
            return np.empty(0)
        return self.model_.predict(X, batch_size=max(self.batch_size, 256), verbose=0).ravel()

    def _save_state(self, path: Path) -> None:
        self.model_.save(path / "model.keras")
        for name, weights in self.weights_.items():
            np.savez(path / f"weights_{name}.npz", *weights)

    def _load_state(self, path: Path) -> None:
        self.model_ = keras.models.load_model(path / "model.keras")
        self.weights_ = {}
        for name in ("best", "last"):
            with np.load(path / f"weights_{name}.npz") as data:
                self.weights_[name] = [data[f"arr_{i}"] for i in range(len(data.files))]
        self.input_shape_ = tuple(self.model_.input_shape[1:])
        self._active_weights = None
        self._use_weights("best")
