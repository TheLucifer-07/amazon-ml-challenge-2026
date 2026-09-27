"""Script 03: Run multi-pass blocking and candidate generation."""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.blocking.candidate_evaluator import evaluate_candidate_recall
from src.blocking.multi_pass_blocking import MultiPassBlocker
from src.data.loader import load_tsv_records
from src.pairs.pair_builder import build_candidate_pairs_df
from src.pairs.pair_validator import validate_candidate_pairs_df
from src.utils.io import load_config
from src.utils.logger import get_logger
from src.utils.timing import timer

logger = get_logger("03_generate_candidates")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate pairs via multi-pass blocking.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--sample-size", type=int, default=10000, help="Number of S1 entities to evaluate (None for full)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    interim_dir = Path(cfg["paths"]["interim_dir"])
    raw_train_dir = Path(cfg["paths"]["raw_train_dir"])
    candidates_dir = Path(cfg["paths"]["candidates_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    candidates_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: CANDIDATE GENERATION & BLOCKING")
    print("=" * 80)

    # Load S1 records
    s1_parquet = interim_dir / "train_source1.parquet"
    if s1_parquet.exists():
        logger.info(f"Loading S1 from {s1_parquet}...")
        s1_df = pd.read_parquet(s1_parquet)
    else:
        s1_df = load_tsv_records(raw_train_dir / "train_source1.tsv", nrows=args.sample_size)

    if args.sample_size and len(s1_df) > args.sample_size:
        s1_df = s1_df.head(args.sample_size).copy()
    logger.info(f"Using {len(s1_df):,} Source 1 entities for candidate generation.")

    # Load S2 and S3 candidates
    cand_dfs = []
    for s_name in ["source2", "source3"]:
        pq_path = interim_dir / f"train_{s_name}.parquet"
        if pq_path.exists():
            logger.info(f"Loading {s_name} from {pq_path}...")
            df = pd.read_parquet(pq_path)
        else:
            df = load_tsv_records(
                raw_train_dir / f"train_{s_name}.tsv", nrows=args.sample_size * 5 if args.sample_size else None
            )
        cand_dfs.append(df)

    candidates_df = pd.concat(cand_dfs, ignore_index=True)
    logger.info(f"Loaded {len(candidates_df):,} total candidate records (S2 + S3).")

    # Initialize and fit multi-pass blocker
    blocker = MultiPassBlocker(max_candidates_per_entity=cfg["blocking"]["max_candidates_per_entity"])
    with timer("Fitting Inverted Index Blockers"):
        blocker.fit(candidates_df)

    # Generate candidates for S1
    with timer("Querying Multi-Pass Candidates for S1"):
        candidate_map = blocker.generate_candidates(s1_df)

    # Load ground truth for recall evaluation
    gt_df = load_tsv_records(raw_train_dir / "train_ground_truth.tsv")
    s1_ids_set = set(s1_df["entity_id"])
    gt_subset = gt_df[gt_df["source1_entity_id"].isin(s1_ids_set)].copy()

    # Evaluate recall and candidate volume
    logger.info("Evaluating candidate retrieval recall against ground truth...")
    eval_report = evaluate_candidate_recall(candidate_map, gt_subset)

    print("\n" + "=" * 80)
    print("CANDIDATE GENERATION RECALL & VOLUME METRICS")
    print("=" * 80)
    print(f"  Candidate Pair Recall:      {eval_report['candidate_pair_recall']:.2%}")
    print(f"  Entities Fully Retrieved:   {eval_report['entity_full_recall']:.2%}")
    print(f"  Total True Pairs:           {eval_report['total_true_pairs']:,}")
    print(f"  Retrieved True Pairs:       {eval_report['retrieved_true_pairs']:,}")
    print(f"  Candidate Volume (Mean):    {eval_report['candidate_volume']['mean']:.1f} per entity")
    print(f"  Candidate Volume (Median):  {eval_report['candidate_volume']['median']:.0f} per entity")
    print(f"  Candidate Volume (P95):     {eval_report['candidate_volume']['p95']:.0f} per entity")
    print(f"  Candidate Volume (Max):     {eval_report['candidate_volume']['max']:,}")

    # Build canonical pairwise dataset
    with timer("Constructing Pairwise DataFrame"):
        pairs_df = build_candidate_pairs_df(candidate_map, s1_df, candidates_df, ground_truth_df=gt_subset)

    val_res = validate_candidate_pairs_df(pairs_df)
    logger.info(f"Candidate pairs validation: is_valid={val_res['is_valid']}, rows={val_res['row_count']:,}")

    # Save report
    rep_path = reports_dir / "candidate_generation_report.json"
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)
    logger.info(f"Saved candidate report to: {rep_path}")

    # Save candidates parquet
    save_parquet = candidates_dir / "train_candidates.parquet"
    pairs_df.to_parquet(save_parquet, index=False)
    logger.info(f"Saved candidate pairs to: {save_parquet}")


if __name__ == "__main__":
    main()
