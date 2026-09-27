"""Business name normalization and legal suffix handling."""

import re

from src.preprocessing.text_normalizer import clean_and_compact, normalize_text_base

LEGAL_SUFFIX_EXPANSIONS = {
    r"\bpvt\b\.?": "private",
    r"\bltd\b\.?": "limited",
    r"\bcorp\b\.?": "corporation",
    r"\binc\b\.?": "incorporated",
    r"\bco\b\.?": "company",
    r"\bllc\b\.?": "limited liability company",
    r"\bllp\b\.?": "limited liability partnership",
    r"\bpllc\b\.?": "professional limited liability company",
    r"\bgmbh\b\.?": "gmbh",
    r"\bsa\b\.?": "sa",
    r"\bsas\b\.?": "sas",
    r"\bsarl\b\.?": "sarl",
    r"\btech\b\.?": "technology",
    r"\btechnologies\b\.?": "technology",
    r"\bintl\b\.?": "international",
    r"\bmfg\b\.?": "manufacturing",
    r"\bserv\b\.?": "services",
    r"\bservices\b\.?": "services",
}

LEGAL_SUFFIXES_REMOVE_PATTERN = re.compile(
    r"\b(private limited|pvt ltd|private|limited|corporation|incorporated|company|corp|inc|llc|llp|gmbh|sa|sas|sarl|ltd|co)\b",
    re.IGNORECASE,
)


def normalize_business_name(name: str | None) -> str:
    """Normalize business name with legal abbreviation expansion."""
    raw = normalize_text_base(name)
    if not raw:
        return ""
    # Standardize & to and
    raw = re.sub(r"&", " and ", raw)
    # Expand legal abbreviations
    for pattern, replacement in LEGAL_SUFFIX_EXPANSIONS.items():
        raw = re.sub(pattern, replacement, raw)
    # Clean punctuation and extra spaces
    return clean_and_compact(raw)


def remove_legal_suffixes(name: str | None) -> str:
    """Remove common legal entity suffixes to isolate the core business brand name."""
    norm = normalize_business_name(name)
    if not norm:
        return ""
    cleaned = LEGAL_SUFFIXES_REMOVE_PATTERN.sub(" ", norm)
    return re.sub(r"\s+", " ", cleaned).strip()


def extract_name_tokens(name: str | None) -> list[str]:
    """Tokenize normalized business name into alphanumeric tokens of length >= 2."""
    norm = normalize_business_name(name)
    if not norm:
        return []
    return [t for t in norm.split() if len(t) >= 2]
