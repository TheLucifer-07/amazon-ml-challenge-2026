"""Unit tests for entity aggregation and singleton handling."""
import pandas as pd

from src.inference.aggregator import aggregate_predictions


def test_aggregate_predictions_singletons_and_matches():
    s1_all = ["S1-001", "S1-002", "S1-003"]
    pairs = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S2-010", "probability": 0.95},
        {"source1_entity_id": "S1-001", "candidate_entity_id": "S3-020", "probability": 0.90},
        {"source1_entity_id": "S1-002", "candidate_entity_id": "S2-099", "probability": 0.40},  # below threshold
    ])

    df_out = aggregate_predictions(s1_all_ids=s1_all, pair_predictions=pairs, threshold=0.70)

    res_map = dict(zip(df_out["source1_entity_id"], df_out["matched_entity_ids"], strict=True))

    assert res_map["S1-001"] == "S2-010,S3-020"
    assert res_map["S1-002"] == ""  # Below threshold -> empty string singleton
    assert res_map["S1-003"] == ""  # Zero candidates -> empty string singleton
    assert len(df_out) == 3
