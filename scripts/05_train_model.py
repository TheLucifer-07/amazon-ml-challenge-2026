"""Script 05: Train Gradient Boosting (LightGBM/XGBoost) model on training candidate pairs."""
import argparse
import json
from pathlib import Path

import pandas as pd

# pyrefly: ignore [missing-import]
from src.models.model_utils import entity_aware_train_val_split

# pyrefly: ignore [missing-import]
from src.models.predict import predict_match_probabilities

# pyrefly: ignore [missing-import]
from src.models.train import train_matching_model

# pyrefly: ignore [missing-import]
from src.utils.io import load_config

# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

# pyrefly: ignore [missing-import]
from src.utils.timing import timer

logger = get_logger("05_train_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train matching ML model.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    features_path = processed_dir / "train_features.parquet"
    logger.info(f"Loading processed features from {features_path}...")
    df = pd.read_parquet(features_path)

    non_feature_cols = ["source1_entity_id", "candidate_entity_id", "label"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Entity-aware split to strictly prevent data leakage
    train_df, val_df = entity_aware_train_val_split(
        df,
        val_size=cfg["evaluation"]["val_size"],
        random_state=cfg["project"]["random_seed"],
    )

    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_val = val_df[feature_cols]


    model_type = cfg["model"].get("type", "lightgbm")
    model_params = {
        "n_estimators": cfg["model"].get("n_estimators", 300),
        "learning_rate": cfg["model"].get("learning_rate", 0.05),
        "max_depth": cfg["model"].get("max_depth", 6),
        "num_leaves": cfg["model"].get("num_leaves", 63),
        "subsample": cfg["model"].get("subsample", 0.8),
        "colsample_bytree": cfg["model"].get("colsample_bytree", 0.8),
        "random_state": cfg["project"]["random_seed"],
        "n_jobs": cfg["model"].get("n_jobs", -1),
    }

    model_save_path = models_dir / "matching_model.joblib"
    with timer(f"Training {model_type} Model"):
        model = train_matching_model(
            X_train,
            y_train,
            model_type=model_type,
            params=model_params,
            save_path=model_save_path,
        )

    # Save feature names
    with open(models_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # Predict probabilities on validation set and save for threshold optimization
    val_probs = predict_match_probabilities(model, X_val)
    val_results = val_df[["source1_entity_id", "candidate_entity_id", "label"]].copy()
    val_results["probability"] = val_probs
    val_results_path = processed_dir / "val_predictions.parquet"
    val_results.to_parquet(val_results_path, index=False)
    logger.info(f"Saved validation predictions to: {val_results_path}")


if __name__ == "__main__":
    main()
