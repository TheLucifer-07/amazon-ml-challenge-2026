"""Output TSV file writer for submission and candidate sets."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("output_builder")


def save_submission_tsv(df_results: pd.DataFrame, output_path: str | Path) -> None:
    """Save final matching results TSV compliant with competition validator rules."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    expected_cols = ["source1_entity_id", "matched_entity_ids"]
    if list(df_results.columns) != expected_cols:
        raise ValueError(f"Invalid columns {list(df_results.columns)}. Expected {expected_cols}")

    logger.info(f"Writing matching results to: {path} ({len(df_results):,} records)")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for row in df_results.itertuples(index=False):
            s1 = str(row.source1_entity_id).strip()
            matches = str(row.matched_entity_ids).strip() if pd.notna(row.matched_entity_ids) else ""
            f.write(f"{s1}\t{matches}\n")


def save_candidate_pairs_tsv(
    s1_all_ids: Sequence[str],
    candidate_map: dict[str, set[str]],
    output_path: str | Path,
) -> None:
    """Save candidate pairs TSV representing the exact candidate set fed into the model."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing candidate pairs to: {path} ({len(s1_all_ids):,} entities)")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1 in s1_all_ids:
            s1_str = str(s1).strip()
            cands = sorted(candidate_map.get(s1_str, set()))
            cand_str = ",".join(cands)
            f.write(f"{s1_str}\t{cand_str}\n")
