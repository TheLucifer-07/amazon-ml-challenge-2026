"""Unit tests for submission generation and validation rules."""

import tempfile
from pathlib import Path

import pandas as pd

from src.inference.output_builder import save_candidate_pairs_tsv, save_submission_tsv


def test_save_submission_tsv():
    df_results = pd.DataFrame(
        [
            {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-005,S3-010"},
            {"source1_entity_id": "S1-002", "matched_entity_ids": ""},
        ]
    )
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv") as f:
        temp_path = f.name

    try:
        save_submission_tsv(df_results, temp_path)
        with open(temp_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert lines[0] == "source1_entity_id\tmatched_entity_ids\n"
        assert lines[1] == "S1-001\tS2-005,S3-010\n"
        assert lines[2] == "S1-002\t\n"  # Singleton: tab present, blank second field
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_save_candidate_pairs_tsv():
    cand_map = {"S1-001": {"S2-005", "S3-010"}, "S1-002": set()}
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv") as f:
        temp_path = f.name

    try:
        save_candidate_pairs_tsv(["S1-001", "S1-002"], cand_map, temp_path)
        with open(temp_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert lines[0] == "source1_entity_id\tcandidate_entity_ids\n"
        assert lines[1] == "S1-001\tS2-005,S3-010\n"
        assert lines[2] == "S1-002\t\n"
    finally:
        Path(temp_path).unlink(missing_ok=True)
