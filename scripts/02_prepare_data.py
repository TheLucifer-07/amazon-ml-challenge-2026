"""Script 02: Prepare and clean data, apply text/address/country normalizations."""

import argparse
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.address_normalizer import normalize_address
from src.preprocessing.country_normalizer import normalize_country
from src.preprocessing.name_normalizer import normalize_business_name, remove_legal_suffixes
from src.utils.io import load_config
from src.utils.logger import get_logger
from src.utils.timing import timer

logger = get_logger("02_prepare_data")

SCHEMA = pa.schema(
    [
        ("entity_id", pa.string()),
        ("business_name", pa.string()),
        ("normalized_name", pa.string()),
        ("clean_name", pa.string()),
        ("business_address", pa.string()),
        ("normalized_address", pa.string()),
        ("country", pa.string()),
        ("normalized_country", pa.string()),
    ]
)


def process_tsv_to_parquet(
    tsv_path: Path,
    parquet_path: Path,
    max_records: int | None = None,
    chunk_size: int = 100_000,
) -> int:
    """Normalize TSV records and stream-write to compressed Parquet."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Normalizing {tsv_path.name} -> {parquet_path.name} (max_records={max_records})...")

    writer = None
    total_processed = 0

    chunk_ids = []
    chunk_names = []
    chunk_norm_names = []
    chunk_clean_names = []
    chunk_addrs = []
    chunk_norm_addrs = []
    chunk_countries = []
    chunk_norm_countries = []

    def flush_chunk():
        nonlocal writer
        batch_dict = {
            "entity_id": chunk_ids,
            "business_name": chunk_names,
            "normalized_name": chunk_norm_names,
            "clean_name": chunk_clean_names,
            "business_address": chunk_addrs,
            "normalized_address": chunk_norm_addrs,
            "country": chunk_countries,
            "normalized_country": chunk_norm_countries,
        }
        table = pa.Table.from_pydict(batch_dict, schema=SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, SCHEMA, compression="zstd")
        writer.write_table(table)
        chunk_ids.clear()
        chunk_names.clear()
        chunk_norm_names.clear()
        chunk_clean_names.clear()
        chunk_addrs.clear()
        chunk_norm_addrs.clear()
        chunk_countries.clear()
        chunk_norm_countries.clear()

    with open(tsv_path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        _header = next(reader, None)

        for row in reader:
            if not row or not row[0].strip():
                continue

            total_processed += 1
            ent_id = row[0].strip()
            name = row[1].strip() if len(row) > 1 else ""
            addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip() if len(row) > 3 else ""

            norm_name = normalize_business_name(name)
            clean_name = remove_legal_suffixes(norm_name)
            norm_addr = normalize_address(addr)
            norm_country = normalize_country(country)

            chunk_ids.append(ent_id)
            chunk_names.append(name)
            chunk_norm_names.append(norm_name)
            chunk_clean_names.append(clean_name)
            chunk_addrs.append(addr)
            chunk_norm_addrs.append(norm_addr)
            chunk_countries.append(country)
            chunk_norm_countries.append(norm_country)

            if len(chunk_ids) >= chunk_size:
                flush_chunk()

            if max_records and total_processed >= max_records:
                break

    if chunk_ids:
        flush_chunk()

    if writer is not None:
        writer.close()

    logger.info(f"Finished {parquet_path.name}: {total_processed:,} records written.")
    return total_processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and clean raw dataset into normalized interim parquet files.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--max-records", type=int, default=None, help="Optional max records limit for testing")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_train_dir = Path(cfg["paths"]["raw_train_dir"])
    raw_test_dir = Path(cfg["paths"]["raw_test_dir"])
    interim_dir = Path(cfg["paths"]["interim_dir"])
    interim_dir.mkdir(parents=True, exist_ok=True)

    files_to_process = [
        (raw_train_dir / "train_source1.tsv", interim_dir / "train_source1.parquet"),
        (raw_train_dir / "train_source2.tsv", interim_dir / "train_source2.parquet"),
        (raw_train_dir / "train_source3.tsv", interim_dir / "train_source3.parquet"),
        (raw_test_dir / "test_source1.tsv", interim_dir / "test_source1.parquet"),
        (raw_test_dir / "test_source2.tsv", interim_dir / "test_source2.parquet"),
        (raw_test_dir / "test_source3.tsv", interim_dir / "test_source3.parquet"),
    ]

    with timer("Normalizing all datasets to Parquet"):
        for tsv_in, parquet_out in files_to_process:
            process_tsv_to_parquet(tsv_in, parquet_out, max_records=args.max_records)


if __name__ == "__main__":
    main()
