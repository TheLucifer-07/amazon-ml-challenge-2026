"""Script 09: Predict matches on test set and output matching_results.tsv and candidate_pairs.tsv."""
import argparse
import json
from pathlib import Path

import pandas as pd

# pyrefly: ignore [missing-import]
from src.blocking.multi_pass_blocking import MultiPassBlocker

# pyrefly: ignore [missing-import]
from src.data.loader import load_tsv_records

# pyrefly: ignore [missing-import]
from src.features.feature_pipeline import build_feature_matrix

# pyrefly: ignore [missing-import]
from src.inference.aggregator import aggregate_predictions

# pyrefly: ignore [missing-import]
from src.inference.output_builder import save_candidate_pairs_tsv, save_submission_tsv

# pyrefly: ignore [missing-import]
from src.models.predict import load_matching_model, predict_match_probabilities

# pyrefly: ignore [missing-import]
from src.pairs.pair_builder import build_candidate_pairs_df

# pyrefly: ignore [missing-import]
from src.utils.io import load_config

# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

# pyrefly: ignore [missing-import]
from src.utils.timing import timer

logger = get_logger("09_predict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict matches on test set.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_test_dir = Path(cfg["paths"]["raw_test_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    thresholds_dir = Path(cfg["paths"]["thresholds_dir"])
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    final_matching_file = Path(cfg["paths"]["final_matching_file"])
    final_candidate_file = Path(cfg["paths"]["final_candidate_file"])
    model_path = models_dir / "matching_model.joblib"
    thresh_path = thresholds_dir / "threshold.json"

    # Load model and threshold
    model = load_matching_model(model_path)
    with open(thresh_path, encoding="utf-8") as f:
        thresh_info = json.load(f)
    best_threshold = thresh_info["best_threshold"]
    logger.info(f"Loaded matching model and decision threshold: {best_threshold:.3f}")

    # Load test datasets
    s1_parquet = interim_dir / "test_source1.parquet"
    if s1_parquet.exists():
        test_s1_df = pd.read_parquet(s1_parquet)
    else:
        test_s1_df = load_tsv_records(raw_test_dir / "test_source1.tsv")

    cand_dfs = []
    for s_name in ["source2", "source3"]:
        pq_path = interim_dir / f"test_{s_name}.parquet"
        if pq_path.exists():
            cand_dfs.append(pd.read_parquet(pq_path))
        else:
            cand_dfs.append(load_tsv_records(raw_test_dir / f"test_{s_name}.tsv"))
    test_cand_df = pd.concat(cand_dfs, ignore_index=True)

    # 1. Blocking & Candidate Generation
    blocker = MultiPassBlocker(max_candidates_per_entity=cfg["blocking"]["max_candidates_per_entity"])
    with timer("Fitting Blocker on Test Candidates"):
        blocker.fit(test_cand_df)

    with timer("Querying Test Candidates for S1"):
        candidate_map = blocker.generate_candidates(test_s1_df)

    s1_all_ids = list(test_s1_df["entity_id"].values)

    # Save candidate_pairs.tsv
    save_candidate_pairs_tsv(s1_all_ids, candidate_map, final_candidate_file)

    # 2. Build Candidate Pairs DataFrame
    with timer("Constructing Test Candidate Pairs"):
        pairs_df = build_candidate_pairs_df(candidate_map, test_s1_df, test_cand_df, ground_truth_df=None)

    if len(pairs_df) > 0:
        # 3. Extract Features
        with timer("Extracting Features for Test Candidates"):
            X_test = build_feature_matrix(pairs_df)

        # Ensure feature alignment
        with open(models_dir / "feature_names.json", encoding="utf-8") as f:
            expected_feats = json.load(f)
        X_test = X_test[expected_feats]

        # 4. Predict Probabilities
        with timer("Predicting Probabilities on Test Pairs"):
            probs = predict_match_probabilities(model, X_test)
            pairs_df["probability"] = probs
    else:
        pairs_df["probability"] = []

    # 5. Aggregate predictions into matching_results.tsv
    with timer("Aggregating Final Entity Predictions"):
        df_matching = aggregate_predictions(s1_all_ids, pairs_df, threshold=best_threshold)
        save_submission_tsv(df_matching, final_matching_file)

    print("\n" + "=" * 80)
    print("TEST INFERENCE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Matching Results Output: {final_matching_file} ({len(df_matching):,} entities)")
    print(f"  Candidate Pairs Output:  {final_candidate_file} ({len(s1_all_ids):,} entities)")


if __name__ == "__main__":
    main()
