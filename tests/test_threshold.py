"""Unit tests for F0.5 optimization and threshold search."""
from src.evaluation.metrics import compute_entity_f_beta
from src.evaluation.threshold_optimizer import optimize_threshold


def test_compute_entity_f_beta_singleton():
    # Singleton correctly predicted empty -> 1.0
    assert compute_entity_f_beta(true_set=set(), pred_set=set(), beta=0.5) == 1.0
    # Singleton falsely merged -> 0.0
    assert compute_entity_f_beta(true_set=set(), pred_set={"S2-001"}, beta=0.5) == 0.0


def test_compute_entity_f_beta_match():
    # True matches: {A, B}, predicted: {A, B, C}
    # Precision = 2/3, Recall = 2/2 = 1.0
    # F0.5 = (1.25 * 2/3 * 1) / (0.25 * 2/3 + 1) = 0.7142857
    score = compute_entity_f_beta(true_set={"A", "B"}, pred_set={"A", "B", "C"}, beta=0.5)
    assert abs(score - 0.7142857) < 1e-4


def test_optimize_threshold():
    gt = {
        "S1-1": {"S2-101"},
        "S1-2": set(),  # singleton
    }
    preds = [
        {"source1_entity_id": "S1-1", "candidate_entity_id": "S2-101", "probability": 0.85},
        {"source1_entity_id": "S1-2", "candidate_entity_id": "S2-999", "probability": 0.30},
    ]
    opt = optimize_threshold(preds, gt, start=0.20, end=0.90, step=0.10)
    assert opt["best_threshold"] > 0.30  # threshold should eliminate S2-999 to score 1.0 on singleton
    assert opt["best_score"] == 1.0
