"""Feature importance extraction and reporting."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("feature_importance")


def extract_feature_importance(
    model: Any,
    feature_names: list[str],
    output_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """Extract and rank model feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        logger.warning("Model does not expose feature_importances_ attribute.")
        return pd.DataFrame()

    df_imp = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values(by="importance", ascending=False)
        .reset_index(drop=True)
    )

    if output_csv_path:
        path = Path(output_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df_imp.to_csv(path, index=False)
        logger.info(f"Saved feature importances to: {path}")

    return df_imp
