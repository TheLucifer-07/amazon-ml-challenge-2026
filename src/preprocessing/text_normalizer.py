"""Centralized base text normalization utilities."""
import re
import unicodedata


def normalize_text_base(text: str | None) -> str:
    """Perform baseline text normalization: Unicode normalization, stripping combining diacritics, lowercase, whitespace trimming."""
    if text is None:
        return ""
    text = str(text)
    # Unicode normalize NFKD and strip combining diacritics (e.g. é -> e)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Convert to lowercase
    text = text.lower()
    # Replace non-breaking spaces and tabs with standard space
    text = re.sub(r"[\s\xa0]+", " ", text)
    return text.strip()



def remove_punctuation(text: str) -> str:
    """Remove standard punctuation characters while preserving alphanumeric and whitespace."""
    return re.sub(r"[^\w\s]", " ", text)


def clean_and_compact(text: str) -> str:
    """Clean punctuation, compact multiple spaces, and strip."""
    text = remove_punctuation(text)
    return re.sub(r"\s+", " ", text).strip()
