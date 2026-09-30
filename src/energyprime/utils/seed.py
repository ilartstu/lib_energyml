"""Reproducibility: one call seeds every random generator that is in use."""
from __future__ import annotations

import os
import random
import sys

import numpy as np


def set_seed(seed: int | None) -> None:
    """Seed Python, NumPy and, if already imported, PyTorch and Keras.

    Heavy frameworks are never imported here: a framework that is not loaded
    yet does not need seeding.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
    if "keras" in sys.modules:
        import keras

        keras.utils.set_random_seed(seed)
