"""Candidate generation evaluation: recall, volume, and reduction ratio."""
from typing import Any

import numpy as np
import pandas as pd


def evaluate_candidate_recall(
    candidate_map: dict[str, set[str]],
    ground_truth_df: pd.DataFrame
) -> dict[str, Any]:
    """Measure candidate retrieval recall against ground truth matches."""
    total_true_pairs = 0
    retrieved_true_pairs = 0
    s1_with_matches = 0
    s1_matches_found = 0

    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        matched_str = str(row["matched_entity_ids"]).strip()
        if not matched_str:
            continue

        true_matches = {m.strip() for m in matched_str.split(",") if m.strip()}
        if not true_matches:
            continue

        s1_with_matches += 1
        candidates = candidate_map.get(s1_id, set())

        found_for_s1 = true_matches.intersection(candidates)
        total_true_pairs += len(true_matches)
        retrieved_true_pairs += len(found_for_s1)

        if len(found_for_s1) == len(true_matches):
            s1_matches_found += 1

    pair_recall = retrieved_true_pairs / total_true_pairs if total_true_pairs > 0 else 0.0
    entity_full_recall = s1_matches_found / s1_with_matches if s1_with_matches > 0 else 0.0

    counts = [len(cands) for cands in candidate_map.values()] if candidate_map else [0]

    return {
        "total_true_pairs": total_true_pairs,
        "retrieved_true_pairs": retrieved_true_pairs,
        "candidate_pair_recall": float(pair_recall),
        "entity_full_recall": float(entity_full_recall),
        "total_s1_entities": len(candidate_map),
        "candidate_volume": {
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "p95": float(np.percentile(counts, 95)),
            "p99": float(np.percentile(counts, 99)),
        }
    }
