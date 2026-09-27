"""Model training pipeline for entity matching."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.model_utils import create_classifier
from src.utils.logger import get_logger

logger = get_logger("model_train")


def train_matching_model(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    model_type: str = "lightgbm",
    params: dict[str, Any] | None = None,
    save_path: str | Path | None = None,
) -> Any:
    """Train gradient boosting matching model and optionally persist artifact."""
    logger.info(f"Training {model_type} classifier on {len(X_train):,} samples...")

    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    logger.info(f"Class distribution: Positive={pos_count:,} ({pos_count / len(y_train):.2%}), Negative={neg_count:,}")

    model = create_classifier(model_type=model_type, params=params)
    model.fit(X_train, y_train)
    logger.info("Model training completed.")

    if save_path:
        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target)
        logger.info(f"Trained model saved to: {target}")

    return model
