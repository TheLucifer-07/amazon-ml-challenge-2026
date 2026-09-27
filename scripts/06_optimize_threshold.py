"""Script 06: Optimize decision threshold for Macro-averaged F0.5 on validation set."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.threshold_optimizer import optimize_threshold
from src.utils.io import load_config
from src.utils.logger import get_logger

logger = get_logger("06_optimize_threshold")


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize threshold for F0.5.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    thresholds_dir = Path(cfg["paths"]["thresholds_dir"])
    thresholds_dir.mkdir(parents=True, exist_ok=True)

    val_preds_path = processed_dir / "val_predictions.parquet"
    logger.info(f"Loading validation predictions from {val_preds_path}...")
    val_df = pd.read_parquet(val_preds_path)

    # Reconstruct ground truth map: {s1_id: set(true_cids)}
    gt_map: dict[str, set[str]] = {}
    for row in val_df.itertuples(index=False):
        s1 = str(row.source1_entity_id)
        if s1 not in gt_map:
            gt_map[s1] = set()
        if getattr(row, "label", 0) == 1:
            gt_map[s1].add(str(row.candidate_entity_id))

    # Convert predictions to list of dicts
    pair_preds = val_df[["source1_entity_id", "candidate_entity_id", "probability"]].to_dict(orient="records")

    t_cfg = cfg["threshold"]
    opt_result = optimize_threshold(
        pair_predictions=pair_preds,
        ground_truth_map=gt_map,
        start=t_cfg.get("start", 0.05),
        end=t_cfg.get("end", 0.95),
        step=t_cfg.get("step", 0.01),
    )

    thresh_save_path = thresholds_dir / "threshold.json"
    with open(thresh_save_path, "w", encoding="utf-8") as f:
        json.dump(opt_result, f, indent=2)
    logger.info(f"Optimal threshold configuration saved to: {thresh_save_path}")

    print("\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"  Selected Optimal Threshold: {opt_result['best_threshold']:.3f}")
    print(f"  Validation Macro F0.5:      {opt_result['best_score']:.4f}")
    print(f"  Precision on Matches:       {opt_result['best_metrics'].get('overall_precision', 0.0):.4f}")
    print(f"  Recall on Matches:          {opt_result['best_metrics'].get('overall_recall', 0.0):.4f}")
    print(f"  Singleton Accuracy:         {opt_result['best_metrics'].get('singletons_accuracy', 0.0):.4f}")


if __name__ == "__main__":
    main()
