"""Official competition evaluation metric: Macro-averaged F0.5 score per Source 1 entity."""

from typing import Any


def compute_entity_f_beta(
    true_set: set[str],
    pred_set: set[str],
    beta: float = 0.5,
) -> float:
    """Compute F-beta for a single Source 1 entity including exact singleton rules."""
    # Singleton case: entity has no true matches
    if not true_set:
        return 1.0 if not pred_set else 0.0

    # Non-singleton case but predicted no match
    if not pred_set:
        return 0.0

    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)

    beta_sq = beta * beta  # 0.25 for beta=0.5
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    if denominator == 0.0:
        return 0.0

    return numerator / denominator


def compute_macro_f05(
    ground_truth_map: dict[str, set[str]],
    predictions_map: dict[str, set[str]],
) -> dict[str, Any]:
    """Compute Macro-averaged F0.5 across all Source 1 entities in ground truth.

    Args:
        ground_truth_map: {s1_id: set(true_match_ids)}
        predictions_map: {s1_id: set(pred_match_ids)}

    Returns:
        dict containing macro_f05, precision_macro, recall_macro, singleton_accuracy, total_entities
    """
    total_entities = len(ground_truth_map)
    if total_entities == 0:
        return {"macro_f05": 0.0, "total_entities": 0}

    scores = []
    singleton_count = 0
    singleton_correct = 0
    match_entities_count = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for s1_id, true_set in ground_truth_map.items():
        pred_set = predictions_map.get(s1_id, set())

        score = compute_entity_f_beta(true_set, pred_set, beta=0.5)
        scores.append(score)

        if not true_set:
            singleton_count += 1
            if not pred_set:
                singleton_correct += 1
            else:
                fp_total += len(pred_set)
        else:
            match_entities_count += 1
            tp = len(true_set.intersection(pred_set))
            tp_total += tp
            fp_total += len(pred_set) - tp
            fn_total += len(true_set) - tp

    macro_f05 = sum(scores) / total_entities
    singleton_acc = singleton_correct / singleton_count if singleton_count > 0 else 1.0

    overall_prec = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    overall_rec = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0

    return {
        "macro_f05": float(macro_f05),
        "total_entities": total_entities,
        "singletons_total": singleton_count,
        "singletons_correct": singleton_correct,
        "singletons_accuracy": float(singleton_acc),
        "entities_with_matches": match_entities_count,
        "pair_tp": tp_total,
        "pair_fp": fp_total,
        "pair_fn": fn_total,
        "overall_precision": float(overall_prec),
        "overall_recall": float(overall_rec),
    }
