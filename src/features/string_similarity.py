"""Edit distance and string similarity feature extractors using RapidFuzz."""

from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.preprocessing.address_normalizer import normalize_address
from src.preprocessing.name_normalizer import normalize_business_name, remove_legal_suffixes


def compute_string_similarity_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
) -> dict[str, float]:
    """Compute normalized edit distance and Jaro-Winkler scores."""
    norm_name_a = normalize_business_name(name_a)
    norm_name_b = normalize_business_name(name_b)

    clean_name_a = remove_legal_suffixes(norm_name_a)
    clean_name_b = remove_legal_suffixes(norm_name_b)

    norm_addr_a = normalize_address(addr_a)
    norm_addr_b = normalize_address(addr_b)

    # Jaro-Winkler similarity [0.0, 1.0]
    name_jw = JaroWinkler.similarity(norm_name_a, norm_name_b) if norm_name_a and norm_name_b else 0.0
    clean_name_jw = JaroWinkler.similarity(clean_name_a, clean_name_b) if clean_name_a and clean_name_b else 0.0
    addr_jw = JaroWinkler.similarity(norm_addr_a, norm_addr_b) if norm_addr_a and norm_addr_b else 0.0

    # Normalized Levenshtein similarity [0.0, 1.0] (1.0 - distance / max_len)
    name_lev = Levenshtein.normalized_similarity(norm_name_a, norm_name_b) if norm_name_a and norm_name_b else 0.0
    clean_name_lev = (
        Levenshtein.normalized_similarity(clean_name_a, clean_name_b) if clean_name_a and clean_name_b else 0.0
    )
    addr_lev = Levenshtein.normalized_similarity(norm_addr_a, norm_addr_b) if norm_addr_a and norm_addr_b else 0.0

    return {
        "name_jaro_winkler": float(name_jw),
        "clean_name_jaro_winkler": float(clean_name_jw),
        "addr_jaro_winkler": float(addr_jw),
        "name_levenshtein": float(name_lev),
        "clean_name_levenshtein": float(clean_name_lev),
        "addr_levenshtein": float(addr_lev),
    }
