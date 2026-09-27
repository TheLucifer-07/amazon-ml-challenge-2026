"""Script 01: Inspect challenge datasets, check schemas, missing values, and report statistics."""
import argparse
import json
from collections import Counter
from pathlib import Path

# pyrefly: ignore [missing-import]
from src.utils.io import load_config

# pyrefly: ignore [missing-import]
from src.utils.logger import get_logger

# pyrefly: ignore [missing-import]
from src.utils.timing import timer

logger = get_logger("01_inspect_data")




def inspect_source_tsv(file_path: Path, is_ground_truth: bool = False) -> tuple[dict, Counter, set[str]]:
    """Stream-inspect a TSV file for memory efficiency and compute exact stats.

    Returns:
        tuple containing:
            - result dict: dataset profile details and statistics
            - id_counts: Counter of entity IDs
            - gt_target_ids: set of matched target IDs (populated only if is_ground_truth=True)
    """
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    logger.info(f"Inspecting {file_path.name} ({file_size_mb:.2f} MB)...")

    row_count = 0
    missing_counts: Counter = Counter()
    country_counts: Counter = Counter()
    id_counts: Counter = Counter()
    sample_rows = []
    gt_target_ids: set[str] = set()

    # Stream line by line to keep memory strictly O(N_ids) without storing full DataFrame
    with open(file_path, encoding="utf-8", errors="replace") as f:
        header_line = f.readline()
        if not header_line:
            err_dict = {"error": "Empty file", "file": str(file_path)}
            return err_dict, Counter(), set()

        headers = [c.strip() for c in header_line.rstrip("\r\n").split("\t")]
        num_cols = len(headers)

        # Ground truth specific counters
        gt_singletons = 0
        gt_match_count_distribution: Counter = Counter()
        gt_s2_matches = 0
        gt_s3_matches = 0

        for line in f:
            line_str = line.rstrip("\r\n")
            if not line_str.strip():
                continue
            row_count += 1
            parts = line_str.split("\t")

            # Pad or truncate if row has mismatched columns
            if len(parts) < num_cols:
                parts += [""] * (num_cols - len(parts))

            # Sample first 3 rows
            if row_count <= 3:
                sample_rows.append(dict(zip(headers, parts[:num_cols], strict=True)))

            # Check missingness
            for col_idx, col_name in enumerate(headers):
                val = parts[col_idx].strip() if col_idx < len(parts) else ""
                if not val:
                    missing_counts[col_name] += 1

            if not is_ground_truth:
                # Entity record
                ent_id = parts[0].strip()
                id_counts[ent_id] += 1

                # Country column (usually 4th column index 3)
                if num_cols >= 4:
                    country_val = parts[3].strip()
                    country_counts[country_val] += 1
            else:
                # Ground truth: source1_entity_id \t matched_entity_ids
                s1_id = parts[0].strip()
                id_counts[s1_id] += 1
                matched_str = parts[1].strip() if len(parts) > 1 else ""

                if not matched_str:
                    gt_singletons += 1
                    gt_match_count_distribution[0] += 1
                else:
                    matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
                    n_matches = len(matched_ids)
                    gt_match_count_distribution[n_matches] += 1
                    for mid in matched_ids:
                        gt_target_ids.add(mid)
                        if mid.startswith("S2-"):
                            gt_s2_matches += 1
                        elif mid.startswith("S3-"):
                            gt_s3_matches += 1

    unique_ids = len(id_counts)
    duplicate_ids = sum(1 for cnt in id_counts.values() if cnt > 1)

    missing_percentages = (
        {k: round(v / row_count * 100, 2) for k, v in missing_counts.items()}
        if row_count
        else {}
    )

    result = {
        "file_name": file_path.name,
        "file_size_mb": round(file_size_mb, 2),
        "columns": headers,
        "column_count": num_cols,
        "row_count": row_count,
        "unique_entity_count": unique_ids,
        "duplicate_id_count": duplicate_ids,
        "missing_counts": dict(missing_counts),
        "missing_percentages": missing_percentages,
        "sample_rows": sample_rows,
    }

    if not is_ground_truth:
        result["country_distribution"] = dict(country_counts)
    else:
        result["ground_truth_stats"] = {
            "total_s1_records": row_count,
            "singletons_count": gt_singletons,
            "singletons_percentage": round(gt_singletons / row_count * 100, 2) if row_count else 0,
            "matched_s1_count": row_count - gt_singletons,
            "match_count_distribution": {f"{k}_matches": v for k, v in sorted(gt_match_count_distribution.items())},
            "total_s2_matches": gt_s2_matches,
            "total_s3_matches": gt_s3_matches,
            "unique_matched_targets": len(gt_target_ids),
        }

    return result, id_counts, gt_target_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect challenge datasets and schemas.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_train_dir = Path(cfg["paths"]["raw_train_dir"])
    raw_test_dir = Path(cfg["paths"]["raw_test_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 1: CHALLENGE DATASET DISCOVERY & INSPECTION")
    print("=" * 80)

    train_files = [
        ("train_source1.tsv", raw_train_dir / "train_source1.tsv", False),
        ("train_source2.tsv", raw_train_dir / "train_source2.tsv", False),
        ("train_source3.tsv", raw_train_dir / "train_source3.tsv", False),
        ("train_ground_truth.tsv", raw_train_dir / "train_ground_truth.tsv", True),
    ]

    test_files = [
        ("test_source1.tsv", raw_test_dir / "test_source1.tsv", False),
        ("test_source2.tsv", raw_test_dir / "test_source2.tsv", False),
        ("test_source3.tsv", raw_test_dir / "test_source3.tsv", False),
    ]

    full_report: dict = {"train": {}, "test": {}, "cross_checks": {}}
    id_pools: dict = {}

    with timer("Inspecting all Training and Test TSVs"):
        for name, path, is_gt in train_files:
            info, id_cnts, targets = inspect_source_tsv(path, is_ground_truth=is_gt)
            full_report["train"][name] = info
            id_pools[name] = (id_cnts, targets)

        for name, path, is_gt in test_files:
            info, id_cnts, targets = inspect_source_tsv(path, is_ground_truth=is_gt)
            full_report["test"][name] = info
            id_pools[name] = (id_cnts, targets)

    # Cross checks on training data
    logger.info("Performing cross-file validation checks...")
    s1_ids = set(id_pools["train_source1.tsv"][0].keys())
    s2_ids = set(id_pools["train_source2.tsv"][0].keys())
    s3_ids = set(id_pools["train_source3.tsv"][0].keys())
    gt_s1_ids = set(id_pools["train_ground_truth.tsv"][0].keys())
    gt_targets = id_pools["train_ground_truth.tsv"][1]

    missing_s1_in_gt = s1_ids - gt_s1_ids
    orphan_gt_s1 = gt_s1_ids - s1_ids
    valid_cand_pool = s2_ids | s3_ids
    invalid_gt_targets = gt_targets - valid_cand_pool

    full_report["cross_checks"]["train"] = {
        "s1_count": len(s1_ids),
        "gt_s1_count": len(gt_s1_ids),
        "s1_ids_missing_from_gt": len(missing_s1_in_gt),
        "gt_ids_not_in_s1": len(orphan_gt_s1),
        "gt_targets_not_in_s2_or_s3": len(invalid_gt_targets),
    }

    # Summary table output
    print("\n" + "=" * 80)
    print("DATASET PROFILE SUMMARY TABLE")
    print("=" * 80)
    header_line = (
        f"{'Dataset / File':<28} | {'Rows':<10} | {'Cols':<6} | "
        f"{'Size MB':<8} | {'Unique IDs':<10} | {'Duplicates':<10}"
    )
    print(header_line)
    print("-" * 80)

    for split in ["train", "test"]:
        for fname, info in full_report[split].items():
            file_label = f"{split}/{fname}"
            row_str = (
                f"{file_label:<28} | {info['row_count']:<10} | {info['column_count']:<6} | "
                f"{info['file_size_mb']:<8} | {info['unique_entity_count']:<10} | {info['duplicate_id_count']:<10}"
            )
            print(row_str)

    print("-" * 80)

    # Print Country Distributions
    print("\nCOUNTRY DISTRIBUTIONS:")
    for split in ["train", "test"]:
        for fname, info in full_report[split].items():
            if "country_distribution" in info:
                print(f"  {split}/{fname}: {info['country_distribution']}")

    # Print Ground Truth Stats
    gt_stats = full_report["train"]["train_ground_truth.tsv"]["ground_truth_stats"]
    print("\nTRAIN GROUND TRUTH MATCHING DYNAMICS:")
    print(f"  Total S1 Entities:     {gt_stats['total_s1_records']:,}")
    print(f"  Singletons (No Match): {gt_stats['singletons_count']:,} ({gt_stats['singletons_percentage']}%)")
    print(f"  Entities with Matches: {gt_stats['matched_s1_count']:,} ({100 - gt_stats['singletons_percentage']:.2f}%)")
    print(f"  Total S2 Matches:      {gt_stats['total_s2_matches']:,}")
    print(f"  Total S3 Matches:      {gt_stats['total_s3_matches']:,}")
    print(f"  Match Count Spread:    {gt_stats['match_count_distribution']}")

    # Save detailed JSON report
    report_json_path = reports_dir / "data_inspection_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nDetailed inspection report saved to: {report_json_path}")


if __name__ == "__main__":
    main()
