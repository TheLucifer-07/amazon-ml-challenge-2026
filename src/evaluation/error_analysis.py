"""Error analysis on false merges (FP) and missed matches (FN)."""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("error_analysis")


def analyze_prediction_errors(
    pairs_df: pd.DataFrame,
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float,
    output_csv_path: str | Path | None = None,
    max_samples: int = 1000,
) -> pd.DataFrame:
    """Identify and catalog False Positives (False Merges) and False Negatives (Missed Matches)."""
    logger.info(f"Running error analysis with threshold={threshold:.3f}...")

    df = pairs_df.copy()
    df["label"] = y_true.to_numpy()
    df["probability"] = y_prob.to_numpy()
    df["prediction"] = (df["probability"] >= threshold).astype(int)

    # Classify outcomes
    df["error_type"] = "TN"
    df.loc[(df["label"] == 1) & (df["prediction"] == 1), "error_type"] = "TP"
    df.loc[(df["label"] == 0) & (df["prediction"] == 1), "error_type"] = "FP"  # False Merge
    df.loc[(df["label"] == 1) & (df["prediction"] == 0), "error_type"] = "FN"  # Missed Match

    fp_count = (df["error_type"] == "FP").sum()
    fn_count = (df["error_type"] == "FN").sum()
    logger.info(f"Error summary: False Merges (FP)={fp_count:,}, Missed Matches (FN)={fn_count:,}")

    errors_df = df[df["error_type"].isin(["FP", "FN"])].copy()

    if output_csv_path:
        path = Path(output_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Sample for CSV export
        sample_export = errors_df.head(max_samples)
        sample_export.to_csv(path, index=False)
        logger.info(f"Saved error analysis sample to: {path}")

    return errors_df
