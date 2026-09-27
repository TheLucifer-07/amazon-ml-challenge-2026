"""Unit tests for candidate pair generation and evaluation."""

import pandas as pd

from src.blocking.candidate_evaluator import evaluate_candidate_recall
from src.pairs.pair_builder import build_candidate_pairs_df


def test_build_candidate_pairs_df():
    cand_map = {"S1-1": {"S2-10", "S3-20"}}
    s1_df = pd.DataFrame(
        [{"entity_id": "S1-1", "business_name": "ABC Co", "business_address": "123 St", "country": "US"}]
    )
    cand_df = pd.DataFrame(
        [
            {"entity_id": "S2-10", "business_name": "ABC", "business_address": "123 St", "country": "US"},
            {"entity_id": "S3-20", "business_name": "XYZ", "business_address": "999 Ave", "country": "US"},
        ]
    )
    gt_df = pd.DataFrame([{"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10"}])

    pairs = build_candidate_pairs_df(cand_map, s1_df, cand_df, gt_df)
    assert len(pairs) == 2
    row1 = pairs[pairs["candidate_entity_id"] == "S2-10"].iloc[0]
    assert row1["label"] == 1
    row2 = pairs[pairs["candidate_entity_id"] == "S3-20"].iloc[0]
    assert row2["label"] == 0


def test_evaluate_candidate_recall():
    cand_map = {"S1-1": {"S2-10", "S2-11"}, "S1-2": set()}
    gt_df = pd.DataFrame(
        [
            {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-10"},
            {"source1_entity_id": "S1-2", "matched_entity_ids": ""},
        ]
    )
    stats = evaluate_candidate_recall(cand_map, gt_df)
    assert stats["candidate_pair_recall"] == 1.0
    assert stats["total_true_pairs"] == 1
