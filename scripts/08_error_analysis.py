"""Script 08: Run error analysis on false merges (FP) and missed matches (FN)."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.error_analysis import analyze_prediction_errors
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger("08_error_analysis")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze false positives and false negatives.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    candidates_dir = Path(cfg["paths"]["candidates_dir"])
    thresholds_dir = Path(cfg["paths"]["thresholds_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])

    val_preds_path = processed_dir / "val_predictions.parquet"
    cand_path = candidates_dir / "train_candidates.parquet"
    thresh_path = thresholds_dir / "threshold.json"
    output_csv = reports_dir / "error_analysis.csv"

    val_df = pd.read_parquet(val_preds_path)
    cand_df = pd.read_parquet(cand_path)

    with open(thresh_path, encoding="utf-8") as f:
        thresh_info = json.load(f)
    best_thresh = thresh_info["best_threshold"]

    # Join metadata with predictions
    merged_df = pd.merge(
        cand_df,
        val_df[["source1_entity_id", "candidate_entity_id", "probability"]],
        on=["source1_entity_id", "candidate_entity_id"],
        how="inner",
    )

    analyze_prediction_errors(
        pairs_df=merged_df,
        y_true=merged_df["label"],
        y_prob=merged_df["probability"],
        threshold=best_thresh,
        output_csv_path=output_csv,
    )


if __name__ == "__main__":
    main()
