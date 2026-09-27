"""Canonical internal schema definitions and type constraints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityRecord:
    """Canonical single entity record representation."""

    entity_id: str
    business_name: str
    business_address: str
    country: str
    source: str  # 'S1', 'S2', or 'S3'


@dataclass(frozen=True)
class CandidatePair:
    """Canonical candidate pair representation between Source 1 and candidate."""

    source1_entity_id: str
    candidate_entity_id: str
    source1_business_name: str
    candidate_business_name: str
    source1_business_address: str
    candidate_business_address: str
    source1_country: str
    candidate_country: str
    label: int | None = None  # 1 for match, 0 for non-match, None during test inference


CANONICAL_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_PAIR_COLUMNS = [
    "source1_entity_id",
    "candidate_entity_id",
    "source1_business_name",
    "candidate_business_name",
    "source1_business_address",
    "candidate_business_address",
    "source1_country",
    "candidate_country",
]
