"""Training history shared by all models (XGBoost rounds, Keras/PyTorch epochs)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class TrainingHistory:
    """Per-step train/validation loss plus the best step.

    ``best_step`` and ``last_step`` are 1-based, as in the notebooks'
    "Best epoch = N" markers.
    """

    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    metric: str = "loss"
    step_name: str = "epoch"
    best_step: int | None = None
    train_time_sec: float | None = None
    stopped_early: bool = False

    @property
    def last_step(self) -> int:
        return len(self.train)

    @property
    def has_val(self) -> bool:
        return len(self.val) > 0

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"train": self.train})
        if self.has_val:
            frame["val"] = self.val
        frame.index = pd.RangeIndex(1, len(frame) + 1, name=self.step_name)
        return frame

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingHistory":
        return cls(**data)

    def __repr__(self) -> str:
        best = f", best {self.step_name} {self.best_step}" if self.best_step else ""
        return f"TrainingHistory({self.last_step} {self.step_name}s{best}, metric={self.metric!r})"
