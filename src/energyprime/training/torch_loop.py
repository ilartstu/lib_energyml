"""Epoch loop for PyTorch models (SVGP): best/last weights, checkpoints, resume.

A model plugs in through :class:`TorchTask`; the trainer does the rest:

* keeps the weights of the best epoch (lowest validation loss) and of the
  last epoch — the model can predict with either;
* optional early stopping (``patience``);
* optional checkpoints in ``checkpoint_dir`` (``last.pt`` every epoch,
  ``best.pt`` on improvement); training continues from them only with
  ``resume=True`` — never silently;
* Ctrl+C / "interrupt kernel" stops training gracefully and keeps progress.
"""
from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from ..core.history import TrainingHistory


class TorchTask:
    """What a PyTorch model provides to :class:`TorchTrainer`."""

    loss_name = "loss"

    def train_mode(self) -> None:
        raise NotImplementedError

    def eval_mode(self) -> None:
        raise NotImplementedError

    def train_step(self, batch: Any) -> float:
        """Forward, backward, optimizer step; return the batch loss."""
        raise NotImplementedError

    def eval_step(self, batch: Any) -> float:
        raise NotImplementedError

    def end_epoch(self) -> None:
        """Called after the training batches of an epoch (e.g. scheduler.step())."""

    def weights(self) -> dict:
        """Copy of the trainable state (for best/last weights)."""
        raise NotImplementedError

    def load_weights(self, state: dict) -> None:
        raise NotImplementedError

    def full_state(self) -> dict:
        """Weights + optimizer + scheduler (for checkpoints)."""
        return {"weights": self.weights()}

    def load_full_state(self, state: dict) -> None:
        self.load_weights(state["weights"])


@dataclass
class TrainingResult:
    history: TrainingHistory
    best_weights: dict
    last_weights: dict


def _progress(iterable: Iterable, verbose: int, total: int):
    if verbose == 1:
        try:
            from tqdm.auto import tqdm

            return tqdm(iterable, total=total, desc="Epochs")
        except ImportError:
            pass
    return iterable


class TorchTrainer:
    def __init__(self, epochs: int = 100, patience: int | None = None, min_delta: float = 0.0,
                 checkpoint_dir: str | Path | None = None, resume: bool = False, verbose: int = 1):
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        self.verbose = verbose

    def fit(self, task: TorchTask, train_loader, val_loader=None) -> TrainingResult:
        ckpt_dir = Path(self.checkpoint_dir) if self.checkpoint_dir else None
        if ckpt_dir:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        start_epoch, train_losses, val_losses = 0, [], []
        best_loss, best_epoch, best_weights = float("inf"), 0, None
        if self.resume:
            if ckpt_dir is None or not (ckpt_dir / "last.pt").exists():
                raise FileNotFoundError("resume=True, but there is no checkpoint_dir/last.pt")
            ckpt = torch.load(ckpt_dir / "last.pt", weights_only=False)
            task.load_full_state(ckpt["state"])
            start_epoch = ckpt["epoch"]
            train_losses, val_losses = ckpt["train_losses"], ckpt["val_losses"]
            best_loss, best_epoch = ckpt["best_loss"], ckpt["best_epoch"]
            if (ckpt_dir / "best.pt").exists():
                best_weights = torch.load(ckpt_dir / "best.pt", weights_only=False)["weights"]

        def save_last(epoch_done: int) -> None:
            if ckpt_dir:
                torch.save({"epoch": epoch_done, "state": task.full_state(), "train_losses": train_losses,
                            "val_losses": val_losses, "best_loss": best_loss, "best_epoch": best_epoch},
                           ckpt_dir / "last.pt")

        wait, stopped_early, epoch_done = 0, False, start_epoch
        t0 = time.perf_counter()
        epochs = _progress(range(start_epoch, self.epochs), self.verbose, self.epochs - start_epoch)
        try:
            for epoch in epochs:
                task.train_mode()
                losses = [task.train_step(batch) for batch in train_loader]
                task.end_epoch()
                train_losses.append(float(np.mean(losses)))
                monitor = train_losses[-1]
                if val_loader is not None:
                    task.eval_mode()
                    with torch.no_grad():
                        val_losses.append(float(np.mean([task.eval_step(batch) for batch in val_loader])))
                    monitor = val_losses[-1]
                epoch_done = epoch + 1
                if monitor < best_loss - self.min_delta:
                    best_loss, best_epoch, wait = monitor, epoch_done, 0
                    best_weights = task.weights()
                    if ckpt_dir:
                        torch.save({"epoch": epoch_done, "weights": best_weights}, ckpt_dir / "best.pt")
                else:
                    wait += 1
                self._log(epochs, epoch_done, train_losses, val_losses)
                save_last(epoch_done)
                if self.patience is not None and wait >= self.patience:
                    stopped_early = True
                    break
        except KeyboardInterrupt:
            warnings.warn(f"Training interrupted after epoch {epoch_done}; progress is kept", stacklevel=2)
            save_last(epoch_done)

        last_weights = task.weights()
        history = TrainingHistory(
            train=train_losses, val=val_losses, metric=task.loss_name, step_name="epoch",
            best_step=best_epoch or None, train_time_sec=time.perf_counter() - t0, stopped_early=stopped_early,
        )
        return TrainingResult(history, best_weights if best_weights is not None else copy.deepcopy(last_weights),
                              last_weights)

    def _log(self, progress, epoch: int, train_losses: list, val_losses: list) -> None:
        text = f"train {train_losses[-1]:.4f}" + (f" | val {val_losses[-1]:.4f}" if val_losses else "")
        if self.verbose == 1 and hasattr(progress, "set_postfix_str"):
            progress.set_postfix_str(text)
        elif self.verbose >= 1:
            print(f"Epoch {epoch}/{self.epochs}: {text}")
