"""Unit tests for feature engineering pipeline."""

import pandas as pd

from src.features.feature_pipeline import build_feature_matrix, extract_pair_features_row


def test_extract_pair_features_row():
    row = {
        "source1_business_name": "Target Store",
        "candidate_business_name": "Target Stores Inc",
        "source1_business_address": "1000 Nicollet Mall, Minneapolis, MN",
        "candidate_business_address": "1000 Nicollet Mall, Minneapolis",
        "source1_country": "US",
        "candidate_country": "USA",
    }
    feats = extract_pair_features_row(row)
    assert "name_jaro_winkler" in feats
    assert "name_jaccard" in feats
    assert "exact_country_match" in feats
    assert feats["exact_country_match"] == 1.0


def test_build_feature_matrix():
    df_pairs = pd.DataFrame(
        [
            {
                "source1_business_name": "Apple Inc",
                "candidate_business_name": "Apple",
                "source1_business_address": "1 Infinite Loop, Cupertino",
                "candidate_business_address": "1 Infinite Loop",
                "source1_country": "US",
                "candidate_country": "US",
            }
        ]
    )
    X = build_feature_matrix(df_pairs)
    assert len(X) == 1
    assert not X.isna().any().any()
