"""Unit tests for string similarity metrics (Jaro-Winkler, Levenshtein, Jaccard)."""
from src.features.exact_match import compute_exact_features
from src.features.string_similarity import compute_string_similarity_features
from src.features.token_similarity import compute_token_features


def test_string_similarity_exact():
    sim = compute_string_similarity_features(
        name_a="Starbucks Coffee",
        name_b="Starbucks Coffee",
        addr_a="123 Main St",
        addr_b="123 Main St",
    )
    assert sim["name_jaro_winkler"] == 1.0
    assert sim["name_levenshtein"] == 1.0
    assert sim["addr_jaro_winkler"] == 1.0


def test_string_similarity_typo():
    sim = compute_string_similarity_features(
        name_a="Starbucks Coffee",
        name_b="Starbuks Cofe",
        addr_a="123 Main St",
        addr_b="123 Mine St",
    )
    assert sim["name_jaro_winkler"] > 0.85
    assert sim["name_levenshtein"] > 0.70


def test_exact_match_features():
    feat = compute_exact_features(
        name_a="Acme Corp",
        name_b="Acme Corporation",
        addr_a="456 Elm Ave",
        addr_b="456 Elm Ave",
        country_a="US",
        country_b="USA",
    )
    assert feat["exact_norm_name_match"] == 1.0  # Corp expands to Corporation
    assert feat["exact_clean_name_match"] == 1.0
    assert feat["exact_country_match"] == 1.0


def test_token_similarity_jaccard():
    tok = compute_token_features(
        name_a="Reliance Retail Limited",
        name_b="Reliance Digital Retail",
        addr_a="MG Road, Bengaluru",
        addr_b="MG Road, Bangalore",
    )
    assert tok["name_shared_tokens"] >= 2.0
    assert tok["name_jaccard"] > 0.4
