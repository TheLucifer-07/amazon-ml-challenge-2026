"""I/O helper utilities for reading configuration, TSV, Parquet, and JSON files."""
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Save serializable data to JSON file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str | Path) -> Any:
    """Load data from JSON file."""
    target = Path(path)
    with open(target, encoding="utf-8") as f:
        return json.load(f)
