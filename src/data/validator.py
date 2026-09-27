"""Validation functions for schema, duplicate records, nulls, and entity IDs."""

from typing import Any

import pandas as pd

from src.data.schema import CANONICAL_COLUMNS


def validate_records_schema(df: pd.DataFrame, expected_prefix: str | None = None) -> dict[str, Any]:
    """Validate DataFrame against canonical record schema and generate profile."""
    errors = []
    warnings = []

    # Check required columns
    missing_cols = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check entity_id string format and prefix
    if "entity_id" in df.columns:
        empty_ids = (df["entity_id"].str.strip() == "").sum()
        if empty_ids > 0:
            errors.append(f"Found {empty_ids} records with empty entity_id")

        if expected_prefix:
            invalid_prefix = (~df["entity_id"].str.startswith(expected_prefix)).sum()
            if invalid_prefix > 0:
                warnings.append(f"Found {invalid_prefix} IDs not starting with {expected_prefix}")

        dup_ids = df["entity_id"].duplicated().sum()
        if dup_ids > 0:
            errors.append(f"Found {dup_ids} duplicate entity IDs")
    else:
        dup_ids = 0

    profile = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": list(df.columns),
        "unique_entity_count": df["entity_id"].nunique() if "entity_id" in df.columns else 0,
        "duplicate_id_count": int(dup_ids),
        "missing_counts": {col: int((df[col].isna() | (df[col].str.strip() == "")).sum()) for col in df.columns},
        "country_distribution": df["country"].value_counts().to_dict() if "country" in df.columns else {},
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }

    return profile
