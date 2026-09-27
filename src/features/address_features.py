"""Address-specific features: postal codes, building numbers, digit patterns."""

import re

from src.preprocessing.address_normalizer import extract_numbers


def compute_address_features(addr_a: str, addr_b: str) -> dict[str, float]:
    """Compute number matching and address component presence features."""
    nums_a = set(extract_numbers(addr_a))
    nums_b = set(extract_numbers(addr_b))

    shared_nums = len(nums_a.intersection(nums_b))
    total_nums = len(nums_a.union(nums_b))
    num_jaccard = shared_nums / total_nums if total_nums > 0 else 0.0

    digits_a = len(re.findall(r"\d", str(addr_a)))
    digits_b = len(re.findall(r"\d", str(addr_b)))
    digit_diff = abs(digits_a - digits_b)

    return {
        "addr_shared_numbers": float(shared_nums),
        "addr_number_jaccard": float(num_jaccard),
        "addr_digit_count_diff": float(digit_diff),
    }
