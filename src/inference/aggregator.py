"""Entity prediction aggregator for final submission formatting."""
from collections import defaultdict
from collections.abc import Sequence

import pandas as pd

# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger("aggregator")


def aggregate_predictions(
    s1_all_ids: Sequence[str],
    pair_predictions: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Aggregate pairwise probabilities into canonical submission format.

    Args:
        s1_all_ids: All Source 1 entity IDs that MUST appear in output.
        pair_predictions: DataFrame with columns:
            ['source1_entity_id', 'candidate_entity_id', 'probability']
        threshold: Decision threshold for accepting a match.

    Returns:
        DataFrame with exactly two columns: ['source1_entity_id', 'matched_entity_ids']
    """
    logger.info(f"Aggregating predictions for {len(s1_all_ids):,} entities with threshold={threshold:.3f}...")

    # Filter accepted matches above threshold
    accepted = pair_predictions[pair_predictions["probability"] >= threshold]

    # Group matches by Source 1 entity
    matches_by_s1: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in accepted.itertuples(index=False):
        s1 = str(row.source1_entity_id)
        cid = str(row.candidate_entity_id)
        prob = float(row.probability)
        matches_by_s1[s1].append((prob, cid))

    rows = []
    singletons = 0
    total_matches = 0

    for s1_id in s1_all_ids:
        s1_str = str(s1_id).strip()
        if s1_str in matches_by_s1:
            # Sort matches by descending probability, then deterministically by ID
            cand_list = matches_by_s1[s1_str]
            cand_list.sort(key=lambda x: (-x[0], x[1]))
            # Remove duplicate candidate IDs while preserving order
            seen_cids = set()
            unique_cids = []
            for _, cid in cand_list:
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    unique_cids.append(cid)

            matched_str = ",".join(unique_cids)
            total_matches += len(unique_cids)
        else:
            matched_str = ""  # Singleton representation
            singletons += 1

        rows.append({"source1_entity_id": s1_str, "matched_entity_ids": matched_str})

    df_out = pd.DataFrame(rows)
    logger.info(
        f"Aggregated {len(df_out):,} rows: {singletons:,} singletons ({singletons/len(df_out):.2%}), "
        f"{total_matches:,} total match links accepted."
    )
    return df_out
