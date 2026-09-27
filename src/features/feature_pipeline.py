"""Pairwise feature engineering pipeline."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.address_features import compute_address_features
from src.features.country_features import compute_country_features
from src.features.exact_match import compute_exact_features
from src.features.missing_features import compute_missing_features
from src.features.string_similarity import compute_string_similarity_features
from src.features.token_similarity import compute_token_features
from src.utils.logger import get_logger

logger = get_logger("feature_pipeline")


def extract_pair_features_row(row_dict: dict[str, Any]) -> dict[str, float]:
    """Extract full numerical feature dictionary for a single candidate pair."""
    name_a = str(row_dict.get("source1_business_name", ""))
    name_b = str(row_dict.get("candidate_business_name", ""))
    addr_a = str(row_dict.get("source1_business_address", ""))
    addr_b = str(row_dict.get("candidate_business_address", ""))
    country_a = str(row_dict.get("source1_country", ""))
    country_b = str(row_dict.get("candidate_country", ""))

    feat: dict[str, float] = {}
    feat.update(compute_exact_features(name_a, name_b, addr_a, addr_b, country_a, country_b))
    feat.update(compute_string_similarity_features(name_a, name_b, addr_a, addr_b))
    feat.update(compute_token_features(name_a, name_b, addr_a, addr_b))
    feat.update(compute_address_features(addr_a, addr_b))
    feat.update(compute_country_features(country_a, country_b))
    feat.update(compute_missing_features(name_a, name_b, addr_a, addr_b))

    return feat


def build_feature_matrix(pairs_df: pd.DataFrame, schema_save_path: str | Path | None = None) -> pd.DataFrame:
    """Transform candidate pairs DataFrame into numerical feature matrix X."""
    logger.info(f"Extracting pairwise similarity features for {len(pairs_df):,} pairs...")

    feature_rows = []
    for row in pairs_df.itertuples(index=False):
        row_dict = row._asdict()
        feat = extract_pair_features_row(row_dict)
        feature_rows.append(feat)

    feat_df = pd.DataFrame(feature_rows)

    # Clean and fill any potential NaNs or Infs
    feat_df = feat_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    feature_names = list(feat_df.columns)
    logger.info(f"Extracted {len(feature_names)} features for {len(feat_df):,} rows.")

    if schema_save_path:
        path = Path(schema_save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        schema_info = {
            "feature_count": len(feature_names),
            "features": feature_names,
            "dtypes": {col: str(feat_df[col].dtype) for col in feature_names},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema_info, f, indent=2)
        logger.info(f"Feature schema saved to: {path}")

    return feat_df
