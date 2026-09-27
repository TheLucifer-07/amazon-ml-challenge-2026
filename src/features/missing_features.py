"""Missingness flags and structural length difference features."""


def compute_missing_features(
    name_a: str,
    name_b: str,
    addr_a: str,
    addr_b: str,
) -> dict[str, float]:
    """Compute presence/absence flags and text length difference features."""
    name_a_clean = str(name_a).strip()
    name_b_clean = str(name_b).strip()
    addr_a_clean = str(addr_a).strip()
    addr_b_clean = str(addr_b).strip()

    addr_a_miss = 1.0 if not addr_a_clean else 0.0
    addr_b_miss = 1.0 if not addr_b_clean else 0.0
    both_addr_miss = 1.0 if addr_a_miss and addr_b_miss else 0.0

    name_len_diff = abs(len(name_a_clean) - len(name_b_clean))
    addr_len_diff = abs(len(addr_a_clean) - len(addr_b_clean))

    return {
        "source_addr_missing": addr_a_miss,
        "candidate_addr_missing": addr_b_miss,
        "both_addr_missing": both_addr_miss,
        "name_char_len_diff": float(name_len_diff),
        "addr_char_len_diff": float(addr_len_diff),
    }
