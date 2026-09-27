"""Decision threshold optimizer for Macro-averaged F0.5."""
from typing import Any

import numpy as np

from src.evaluation.metrics import compute_macro_f05
from src.utils.logger import get_logger

logger = get_logger("threshold_optimizer")


def optimize_threshold(
    pair_predictions: list[dict[str, Any]],
    ground_truth_map: dict[str, set[str]],
    start: float = 0.05,
    end: float = 0.95,
    step: float = 0.01,
) -> dict[str, Any]:
    """Sweep threshold values to find the threshold yielding maximum Macro F0.5.

    Args:
        pair_predictions: list of {'source1_entity_id': s1, 'candidate_entity_id': cid, 'probability': prob}
        ground_truth_map: {s1_id: set(matched_cids)}
    """
    logger.info(f"Sweeping thresholds from {start:.2f} to {end:.2f} (step {step:.2f})...")
    thresholds = np.arange(start, end + 1e-5, step)

    best_threshold = 0.50
    best_score = -1.0
    best_metrics = {}
    history = []

    for thresh in thresholds:
        t = round(float(thresh), 3)

        # Aggregate predicted IDs above threshold
        preds_map: dict[str, set[str]] = {s1: set() for s1 in ground_truth_map}
        for item in pair_predictions:
            s1 = item["source1_entity_id"]
            if s1 in preds_map and item["probability"] >= t:
                preds_map[s1].add(item["candidate_entity_id"])

        res = compute_macro_f05(ground_truth_map, preds_map)
        score = res["macro_f05"]
        history.append({"threshold": t, "macro_f05": score, "precision": res["overall_precision"], "recall": res["overall_recall"]})

        if score > best_score:
            best_score = score
            best_threshold = t
            best_metrics = res

    logger.info(f"Optimal Threshold: {best_threshold:.3f} | Best Macro F0.5: {best_score:.4f}")
    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "sweep_history": history,
    }
