"""Vectorized inverted index candidate generator for ultra-fast entity blocking."""
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("candidate_generator")


class InvertedIndexBlocker:
    """Builds an inverted index of blocking keys over candidate datasets (S2, S3) and queries matches for S1."""

    def __init__(self, key_extractor_func: Callable[[dict[str, Any]], Any], name: str = "blocker"):
        self.key_extractor = key_extractor_func
        self.name = name
        self.index: dict[str, set[str]] = defaultdict(set)

    def fit_from_arrays(self, keys: Sequence[str], cids: Sequence[str]) -> "InvertedIndexBlocker":
        """Index candidate records directly from precomputed key and ID arrays in seconds."""
        logger.info(f"Building index '{self.name}' on {len(cids):,} precomputed keys...")
        self.index.clear()
        for cid, k in zip(cids, keys, strict=True):
            if k:
                self.index[k].add(cid)
        logger.info(f"Index '{self.name}' built with {len(self.index):,} distinct keys.")
        return self

    def fit(self, candidate_df: pd.DataFrame) -> "InvertedIndexBlocker":
        """Fallback fit method iterating rows."""
        logger.info(f"Building index '{self.name}' on {len(candidate_df):,} candidate records...")
        self.index.clear()
        for row in candidate_df.itertuples(index=False):
            row_dict = row._asdict()
            cid = row_dict["entity_id"]
            keys = self.key_extractor(row_dict)
            if isinstance(keys, str):
                keys = [keys] if keys else []
            for k in keys:
                if k:
                    self.index[k].add(cid)
        return self

    def query(self, s1_record: dict[str, Any] | pd.Series) -> set[str]:
        """Query candidate IDs matching keys of a single S1 record."""
        record_dict = s1_record.to_dict() if hasattr(s1_record, "to_dict") else s1_record
        keys = self.key_extractor(record_dict)
        if isinstance(keys, str):
            keys = [keys] if keys else []

        candidates: set[str] = set()
        for k in keys:
            if k in self.index:
                candidates.update(self.index[k])
        return candidates
