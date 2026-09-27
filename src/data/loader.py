import csv
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_loader")


def load_tsv_records(file_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load challenge TSV file preserving entity_id as string with leading zeros."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    logger.info(f"Loading TSV: {path} (nrows={nrows})")
    df = pd.read_csv(
        path,
        sep="\t",
        dtype={
            "entity_id": str,
            "business_name": str,
            "business_address": str,
            "country": str,
            "source1_entity_id": str,
            "matched_entity_ids": str,
            "candidate_entity_ids": str,
        },
        keep_default_na=False,
        nrows=nrows,
        encoding="utf-8",
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    return df
