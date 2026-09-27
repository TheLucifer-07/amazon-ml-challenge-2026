"""Open-set country normalization and matching."""

import re

from src.preprocessing.text_normalizer import normalize_text_base

KNOWN_COUNTRY_VARIANTS = {
    "us": "us",
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "in": "india",
    "ind": "india",
    "india": "india",
    "fr": "france",
    "fra": "france",
    "france": "france",
}


def normalize_country(country: str | None) -> str:
    """Normalize country string in an open-set fashion without crashing on unseen countries."""
    raw = normalize_text_base(country)
    if not raw:
        return "unknown"
    # Check known aliases
    if raw in KNOWN_COUNTRY_VARIANTS:
        return KNOWN_COUNTRY_VARIANTS[raw]
    # For any unseen country (e.g. Germany, Japan, France), clean punctuation and return
    cleaned = re.sub(r"[^\w\s]", "", raw).strip()
    return cleaned if cleaned else "unknown"


def is_country_match(country_a: str | None, country_b: str | None) -> bool:
    """Check whether two country strings represent the same country."""
    norm_a = normalize_country(country_a)
    norm_b = normalize_country(country_b)
    if norm_a == "unknown" or norm_b == "unknown":
        return False
    return norm_a == norm_b
