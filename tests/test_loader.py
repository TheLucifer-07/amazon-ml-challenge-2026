"""Unit tests for data loader and schema."""
import tempfile
from pathlib import Path

from src.data.loader import load_tsv_records
from src.data.validator import validate_records_schema


def test_load_tsv_records_preserves_strings():
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".tsv") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("001234\tTest Shop\t100 Main St\tUS\n")
        temp_path = f.name

    try:
        df = load_tsv_records(temp_path)
        assert df["entity_id"].iloc[0] == "001234"  # Leading zero preserved as string!
        profile = validate_records_schema(df)
        assert profile["is_valid"] is True
        assert profile["row_count"] == 1
    finally:
        Path(temp_path).unlink(missing_ok=True)
