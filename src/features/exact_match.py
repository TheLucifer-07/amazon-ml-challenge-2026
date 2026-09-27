"""Exact match feature extractors for entity resolution."""
# pyrefly: ignore [missing-import]
from src.preprocessing.address_normalizer import normalize_address

# pyrefly: ignore [missing-import]
from src.preprocessing.country_normalizer import is_country_match

# pyrefly: ignore [missing-import]
from src.preprocessing.name_normalizer import normalize_business_name, remove_legal_suffixes


def compute_exact_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
    country_a: str,
    country_b: str,
) -> dict[str, float]:
    """Compute exact and normalized exact equality signals."""
    norm_name_a = normalize_business_name(name_a)
    norm_name_b = normalize_business_name(name_b)

    clean_name_a = remove_legal_suffixes(norm_name_a)
    clean_name_b = remove_legal_suffixes(norm_name_b)

    norm_addr_a = normalize_address(addr_a)
    norm_addr_b = normalize_address(addr_b)

    return {
        "exact_raw_name_match": 1.0 if name_a and name_a.strip().lower() == name_b.strip().lower() else 0.0,
        "exact_norm_name_match": 1.0 if norm_name_a and norm_name_a == norm_name_b else 0.0,
        "exact_clean_name_match": 1.0 if clean_name_a and clean_name_a == clean_name_b else 0.0,
        "exact_raw_addr_match": 1.0 if addr_a and addr_a.strip().lower() == addr_b.strip().lower() else 0.0,
        "exact_norm_addr_match": 1.0 if norm_addr_a and norm_addr_a == norm_addr_b else 0.0,
        "exact_country_match": 1.0 if is_country_match(country_a, country_b) else 0.0,
    }
