"""Model inference and probability prediction."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_matching_model(model_path: str | Path) -> Any:
    """Load persisted matching model artifact."""
    target = Path(model_path)
    if not target.exists():
        raise FileNotFoundError(f"Model file not found: {target}")
    return joblib.load(target)


def predict_match_probabilities(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Predict continuous probability of candidate pair being a match: P(same_entity)."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = model.predict(X)
    return np.asarray(probs, dtype=np.float32)
