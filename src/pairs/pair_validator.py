"""Validator for canonical candidate-pair datasets."""

from typing import Any

import pandas as pd

from src.data.schema import REQUIRED_PAIR_COLUMNS


def validate_candidate_pairs_df(df: pd.DataFrame) -> dict[str, Any]:
    """Validate candidate pairs dataframe schema and constraints."""
    errors = []
    missing_cols = [c for c in REQUIRED_PAIR_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns in candidate pairs: {missing_cols}")

    if "source1_entity_id" in df.columns and "candidate_entity_id" in df.columns:
        empty_s1 = (df["source1_entity_id"].astype(str).str.strip() == "").sum()
        empty_cand = (df["candidate_entity_id"].astype(str).str.strip() == "").sum()
        if empty_s1 > 0 or empty_cand > 0:
            errors.append(f"Found empty entity IDs: S1={empty_s1}, Cand={empty_cand}")

        dup_pairs = df.duplicated(subset=["source1_entity_id", "candidate_entity_id"]).sum()
        if dup_pairs > 0:
            errors.append(f"Found {dup_pairs} duplicate candidate pairs")
    else:
        dup_pairs = 0

    has_labels = "label" in df.columns
    label_dist = df["label"].value_counts().to_dict() if has_labels else {}

    return {
        "row_count": len(df),
        "columns": list(df.columns),
        "is_valid": len(errors) == 0,
        "duplicate_pairs": int(dup_pairs),
        "has_labels": has_labels,
        "label_distribution": label_dist,
        "errors": errors,
    }
