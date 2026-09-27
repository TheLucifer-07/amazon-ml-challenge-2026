"""Script 04: Compute pairwise similarity features for candidate pairs."""
import argparse
from pathlib import Path

import pandas as pd

from src.features.feature_pipeline import build_feature_matrix
from src.utils.io import load_config
from src.utils.logger import get_logger
from src.utils.timing import timer

logger = get_logger("04_build_features")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pairwise similarity features.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input-file", default=None, help="Input candidate parquet file")
    parser.add_argument("--output-file", default=None, help="Output features parquet file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    candidates_dir = Path(cfg["paths"]["candidates_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    schemas_dir = Path(cfg["paths"]["schemas_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    schemas_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input_file) if args.input_file else (candidates_dir / "train_candidates.parquet")
    output_path = Path(args.output_file) if args.output_file else (processed_dir / "train_features.parquet")
    schema_path = schemas_dir / "feature_schema.json"

    logger.info(f"Loading candidate pairs from: {input_path}...")
    pairs_df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(pairs_df):,} candidate pairs.")

    with timer("Extracting Pairwise Feature Matrix"):
        X_df = build_feature_matrix(pairs_df, schema_save_path=schema_path)

    # Attach labels and identifier columns for downstream training and evaluation
    if "label" in pairs_df.columns:
        X_df["label"] = pairs_df["label"].values
    X_df["source1_entity_id"] = pairs_df["source1_entity_id"].values
    X_df["candidate_entity_id"] = pairs_df["candidate_entity_id"].values

    X_df.to_parquet(output_path, index=False)
    logger.info(f"Saved feature matrix to: {output_path} ({len(X_df):,} rows, {len(X_df.columns)} columns)")


if __name__ == "__main__":
    main()
