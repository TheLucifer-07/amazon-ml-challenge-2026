"""Business address normalization without external geocoding lookup."""

import re

from src.preprocessing.text_normalizer import clean_and_compact, normalize_text_base

ADDRESS_ABBREVIATIONS = {
    r"\brd\b\.?": "road",
    r"\bst\b\.?": "street",
    r"\bave\b\.?": "avenue",
    r"\bblvd\b\.?": "boulevard",
    r"\bdr\b\.?": "drive",
    r"\bln\b\.?": "lane",
    r"\bhwy\b\.?": "highway",
    r"\bpkwy\b\.?": "parkway",
    r"\bct\b\.?": "court",
    r"\bpl\b\.?": "place",
    r"\bsq\b\.?": "square",
    r"\bste\b\.?": "suite",
    r"\bapt\b\.?": "apartment",
    r"\bfl\b\.?": "floor",
    r"\bbldg\b\.?": "building",
    r"\bdept\b\.?": "department",
    r"\bctr\b\.?": "center",
    r"\bpo box\b\.?": "pobox",
    r"\bp\.o\. box\b": "pobox",
    r"\bnear\b": "near",
    r"\bopp\b\.?": "opposite",
}


def normalize_address(address: str | None) -> str:
    """Normalize address string using purely local heuristic expansion."""
    raw = normalize_text_base(address)
    if not raw:
        return ""
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        raw = re.sub(pattern, replacement, raw)
    return clean_and_compact(raw)


def extract_address_tokens(address: str | None) -> list[str]:
    """Extract distinct address tokens."""
    norm = normalize_address(address)
    if not norm:
        return []
    return [t for t in norm.split() if len(t) >= 2]


def extract_numbers(address: str | None) -> list[str]:
    """Extract sequences of digits (postal codes, building/suite numbers)."""
    if not address:
        return []
    return re.findall(r"\b\d+\b", str(address))
