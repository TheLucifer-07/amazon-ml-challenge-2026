"""Vectorized multi-pass blocking orchestration and candidate pair generation."""
import pandas as pd

# pyrefly: ignore [missing-import]
from src.blocking.candidate_generator import InvertedIndexBlocker

# pyrefly: ignore [missing-import]
from src.preprocessing.country_normalizer import normalize_country

# pyrefly: ignore [missing-import]
from src.preprocessing.name_normalizer import normalize_business_name, remove_legal_suffixes

# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

logger = get_logger("multi_pass_blocking")


class MultiPassBlocker:
    """Combines multiple blocking strategies (exact, clean, prefix) to maximize recall."""

    def __init__(self, max_candidates_per_entity: int = 100):
        self.max_candidates = max_candidates_per_entity
        self.blocker_exact = InvertedIndexBlocker(lambda r: "", name="pass1_exact_name")
        self.blocker_clean = InvertedIndexBlocker(lambda r: "", name="pass2_clean_name")
        self.blocker_prefix = InvertedIndexBlocker(lambda r: "", name="pass3_prefix_4")

    def _ensure_normalized_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure normalized columns exist, computing vectorially if needed."""
        df_out = df.copy()
        if "normalized_country" not in df_out.columns:
            df_out["normalized_country"] = df_out["country"].apply(normalize_country)
        if "normalized_name" not in df_out.columns:
            df_out["normalized_name"] = df_out["business_name"].apply(normalize_business_name)
        if "clean_name" not in df_out.columns:
            df_out["clean_name"] = df_out["normalized_name"].apply(remove_legal_suffixes)
        return df_out

    def fit(self, candidate_df: pd.DataFrame) -> "MultiPassBlocker":
        """Fit all blockers vectorially on candidates in seconds."""
        logger.info(f"Vectorizing keys for {len(candidate_df):,} candidate records...")
        cand_df = self._ensure_normalized_cols(candidate_df)

        cids = cand_df["entity_id"].values
        norm_country = cand_df["normalized_country"].astype(str)
        norm_name = cand_df["normalized_name"].astype(str)
        clean_name = cand_df["clean_name"].astype(str)

        # Pass 1: exact normalized name
        keys_exact = (norm_country + "::" + norm_name).values
        self.blocker_exact.fit_from_arrays(keys_exact, cids)

        # Pass 2: clean name (stripped of legal suffixes)
        keys_clean = (norm_country + "::" + clean_name).values
        self.blocker_clean.fit_from_arrays(keys_clean, cids)

        # Pass 3: 4-character prefix of clean name
        clean_no_space = clean_name.str.replace(" ", "", regex=False).str.slice(0, 4)
        keys_prefix = (norm_country + "::" + clean_no_space).values
        self.blocker_prefix.fit_from_arrays(keys_prefix, cids)

        logger.info("All blocking indexes fitted successfully.")
        return self

    def generate_candidates(self, s1_df: pd.DataFrame) -> dict[str, set[str]]:
        """Generate candidate sets for all S1 entities via fast dictionary queries."""
        logger.info(f"Generating candidate pairs for {len(s1_df):,} Source 1 entities...")
        s1_prep = self._ensure_normalized_cols(s1_df)

        cids = s1_prep["entity_id"].values
        norm_country = s1_prep["normalized_country"].astype(str)
        norm_name = s1_prep["normalized_name"].astype(str)
        clean_name = s1_prep["clean_name"].astype(str)

        keys_exact = (norm_country + "::" + norm_name).values
        keys_clean = (norm_country + "::" + clean_name).values
        clean_no_space = clean_name.str.replace(" ", "", regex=False).str.slice(0, 4)
        keys_prefix = (norm_country + "::" + clean_no_space).values

        results: dict[str, set[str]] = {}

        for s1_id, k_exact, k_clean, k_pfx in zip(cids, keys_exact, keys_clean, keys_prefix, strict=True):
            candidates: set[str] = set()

            # Pass 1: Exact name key
            if k_exact in self.blocker_exact.index:
                candidates.update(self.blocker_exact.index[k_exact])

            # Pass 2: Clean name key
            if len(candidates) < self.max_candidates and k_clean in self.blocker_clean.index:
                candidates.update(self.blocker_clean.index[k_clean])

            # Pass 3: Prefix key (selective, avoid generic explosion)
            if len(candidates) < 20 and k_pfx in self.blocker_prefix.index:
                pfx_cands = self.blocker_prefix.index[k_pfx]
                if len(pfx_cands) <= 50:
                    candidates.update(pfx_cands)

            results[s1_id] = candidates

        return results
