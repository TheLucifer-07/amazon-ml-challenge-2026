"""Open-set country relationship features."""
# pyrefly: ignore [missing-import]
from src.preprocessing.country_normalizer import is_country_match, normalize_country


def compute_country_features(country_a: str, country_b: str) -> dict[str, float]:
    """Compute country matching features without hardcoding closed country lists."""
    norm_a = normalize_country(country_a)
    norm_b = normalize_country(country_b)

    return {
        "country_match": 1.0 if is_country_match(country_a, country_b) else 0.0,
        "country_is_us": 1.0 if norm_a == "us" and norm_b == "us" else 0.0,
        "country_is_india": 1.0 if norm_a == "india" and norm_b == "india" else 0.0,
        "country_is_france": 1.0 if norm_a == "france" and norm_b == "france" else 0.0,
    }
