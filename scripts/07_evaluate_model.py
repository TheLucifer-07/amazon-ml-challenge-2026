"""Script 07: Comprehensive model evaluation on validation set."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.feature_importance import extract_feature_importance
from src.evaluation.metrics import compute_macro_f05
from src.models.predict import load_matching_model
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger("07_evaluate_model")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate matching model on validation set.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    thresholds_dir = Path(cfg["paths"]["thresholds_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    val_preds_path = processed_dir / "val_predictions.parquet"
    thresh_path = thresholds_dir / "threshold.json"
    model_path = models_dir / "matching_model.joblib"
    feat_names_path = models_dir / "feature_names.json"

    val_df = pd.read_parquet(val_preds_path)
    with open(thresh_path, encoding="utf-8") as f:
        thresh_info = json.load(f)
    best_threshold = thresh_info["best_threshold"]

    # Reconstruct ground truth map & predictions map
    gt_map: dict[str, set[str]] = {}
    preds_map: dict[str, set[str]] = {}

    for row in val_df.itertuples(index=False):
        s1 = str(row.source1_entity_id)
        if s1 not in gt_map:
            gt_map[s1] = set()
            preds_map[s1] = set()
        if getattr(row, "label", 0) == 1:
            gt_map[s1].add(str(row.candidate_entity_id))
        if getattr(row, "probability", 0.0) >= best_threshold:
            preds_map[s1].add(str(row.candidate_entity_id))

    metrics = compute_macro_f05(gt_map, preds_map)

    # Feature importances
    if model_path.exists() and feat_names_path.exists():
        model = load_matching_model(model_path)
        with open(feat_names_path, encoding="utf-8") as f:
            feat_names = json.load(f)
        extract_feature_importance(model, feat_names, output_csv_path=reports_dir / "feature_importance.csv")

    metrics_save_path = reports_dir / "validation_metrics.json"
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved validation metrics to: {metrics_save_path}")

    print("\n" + "=" * 80)
    print("FINAL VALIDATION EVALUATION REPORT")
    print("=" * 80)
    print(f"  Macro-averaged F0.5:      {metrics['macro_f05']:.4f}")
    print(f"  Precision:                 {metrics['overall_precision']:.4f}")
    print(f"  Recall:                    {metrics['overall_recall']:.4f}")
    print(f"  Singleton Accuracy:        {metrics['singletons_accuracy']:.4f} ({metrics['singletons_correct']}/{metrics['singletons_total']})")
    print(f"  Pair True Positives (TP):  {metrics['pair_tp']:,}")
    print(f"  Pair False Positives (FP): {metrics['pair_fp']:,}")
    print(f"  Pair False Negatives (FN): {metrics['pair_fn']:,}")


if __name__ == "__main__":
    main()
