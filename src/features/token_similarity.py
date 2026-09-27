"""Token-level overlap and Jaccard similarity features."""
# pyrefly: ignore [missing-import]
from src.preprocessing.address_normalizer import extract_address_tokens

# pyrefly: ignore [missing-import]
from src.preprocessing.name_normalizer import extract_name_tokens


def compute_token_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
) -> dict[str, float]:
    """Compute token intersection, union, overlap ratio, and Jaccard similarity."""
    tok_name_a = set(extract_name_tokens(name_a))
    tok_name_b = set(extract_name_tokens(name_b))

    tok_addr_a = set(extract_address_tokens(addr_a))
    tok_addr_b = set(extract_address_tokens(addr_b))

    # Name token overlap
    name_inter = len(tok_name_a.intersection(tok_name_b))
    name_union = len(tok_name_a.union(tok_name_b))
    name_jaccard = name_inter / name_union if name_union > 0 else 0.0
    min_name_tokens = min(len(tok_name_a), len(tok_name_b))
    name_overlap_ratio = name_inter / min_name_tokens if min_name_tokens > 0 else 0.0

    # Address token overlap
    addr_inter = len(tok_addr_a.intersection(tok_addr_b))
    addr_union = len(tok_addr_a.union(tok_addr_b))
    addr_jaccard = addr_inter / addr_union if addr_union > 0 else 0.0
    min_addr_tokens = min(len(tok_addr_a), len(tok_addr_b))
    addr_overlap_ratio = addr_inter / min_addr_tokens if min_addr_tokens > 0 else 0.0

    # Length ratios
    len_diff = abs(len(tok_name_a) - len(tok_name_b))
    max_len = max(len(tok_name_a), len(tok_name_b))
    len_ratio = len_diff / max_len if max_len > 0 else 0.0

    return {
        "name_jaccard": float(name_jaccard),
        "name_shared_tokens": float(name_inter),
        "name_overlap_ratio": float(name_overlap_ratio),
        "addr_jaccard": float(addr_jaccard),
        "addr_shared_tokens": float(addr_inter),
        "addr_overlap_ratio": float(addr_overlap_ratio),
        "name_token_count_diff": float(len_diff),
        "name_token_len_ratio": float(len_ratio),
    }
