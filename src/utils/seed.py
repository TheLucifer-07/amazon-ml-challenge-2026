"""Seed utility to ensure exact reproducibility across all stochastic components."""
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set seeds for Python random, numpy, and environment variables."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
