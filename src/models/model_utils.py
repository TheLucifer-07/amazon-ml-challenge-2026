"""Model initialization, entity-aware splitting, and hyperparameter utilities."""
from typing import Any

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit

from src.utils.logger import get_logger

logger = get_logger("model_utils")


def entity_aware_train_val_split(
    pairs_df: pd.DataFrame,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candidate pairs by source1_entity_id to strictly prevent entity-level leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    groups = pairs_df["source1_entity_id"]

    train_idx, val_idx = next(gss.split(pairs_df, groups=groups))
    train_df = pairs_df.iloc[train_idx].reset_index(drop=True)
    val_df = pairs_df.iloc[val_idx].reset_index(drop=True)

    # Double-check zero overlap between entity groups
    train_entities = set(train_df["source1_entity_id"].unique())
    val_entities = set(val_df["source1_entity_id"].unique())
    overlap = train_entities.intersection(val_entities)
    if overlap:
        raise ValueError(f"Entity leakage detected! {len(overlap)} entities in both train and val.")

    logger.info(
        f"Entity-aware split: {len(train_df):,} train pairs ({len(train_entities):,} entities), "
        f"{len(val_df):,} val pairs ({len(val_entities):,} entities)."
    )
    return train_df, val_df


def create_classifier(model_type: str = "lightgbm", params: dict[str, Any] | None = None) -> Any:
    """Instantiate Gradient Boosting Classifier (LightGBM or XGBoost)."""
    default_params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    if model_type.lower() == "lightgbm":
        return lgb.LGBMClassifier(**default_params)
    elif model_type.lower() == "xgboost":
        return xgb.XGBClassifier(**default_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'lightgbm' or 'xgboost'.")
