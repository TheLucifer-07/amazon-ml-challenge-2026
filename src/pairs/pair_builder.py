"""Build canonical candidate pairs from Source 1 and candidate datasets."""
import pandas as pd

# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger("pair_builder")


def build_candidate_pairs_df(
    candidate_map: dict[str, set[str]],
    s1_df: pd.DataFrame,
    candidate_records_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join candidate IDs with entity records to construct canonical pairwise dataset with O(1) dict lookups."""
    logger.info(f"Building candidate pairs dataframe for {len(candidate_map):,} Source 1 entities...")

    # Fast O(1) lookups: only index candidate IDs that are actually referenced
    s1_lookup = s1_df.set_index("entity_id").to_dict(orient="index")
    needed_cand_ids = set().union(*candidate_map.values()) if candidate_map else set()
    cand_sub = candidate_records_df[candidate_records_df["entity_id"].isin(needed_cand_ids)]
    cand_lookup = cand_sub.set_index("entity_id").to_dict(orient="index")

    # Build ground truth set of positive tuples (s1_id, match_id)
    positive_pairs = set()
    if ground_truth_df is not None:
        for row in ground_truth_df.itertuples(index=False):
            row_dict = row._asdict()
            s1_id = str(row_dict.get("source1_entity_id", ""))
            m_str = str(row_dict.get("matched_entity_ids", "")).strip()
            if m_str:
                for cid in m_str.split(","):
                    c_clean = cid.strip()
                    if c_clean:
                        positive_pairs.add((s1_id, c_clean))

    rows = []
    for s1_id, cands in candidate_map.items():
        if s1_id not in s1_lookup:
            continue
        s1_info = s1_lookup[s1_id]

        for cid in cands:
            if cid not in cand_lookup:
                continue
            cand_info = cand_lookup[cid]

            pair_dict = {
                "source1_entity_id": s1_id,
                "candidate_entity_id": cid,
                "source1_business_name": s1_info["business_name"],
                "candidate_business_name": cand_info["business_name"],
                "source1_business_address": s1_info["business_address"],
                "candidate_business_address": cand_info["business_address"],
                "source1_country": s1_info["country"],
                "candidate_country": cand_info["country"],
            }
            if ground_truth_df is not None:
                pair_dict["label"] = 1 if (s1_id, cid) in positive_pairs else 0

            rows.append(pair_dict)

    df_pairs = pd.DataFrame(rows)
    logger.info(f"Built {len(df_pairs):,} candidate pairs.")
    return df_pairs
