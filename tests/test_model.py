"""Unit tests for ML models (LightGBM/XGBoost)."""

import numpy as np
import pandas as pd

from src.models.model_utils import entity_aware_train_val_split
from src.models.predict import predict_match_probabilities
from src.models.train import train_matching_model


def test_entity_aware_split_no_leakage():
    df = pd.DataFrame(
        [
            {"source1_entity_id": "S1-A", "candidate_entity_id": "S2-1", "label": 1},
            {"source1_entity_id": "S1-A", "candidate_entity_id": "S2-2", "label": 0},
            {"source1_entity_id": "S1-B", "candidate_entity_id": "S2-3", "label": 1},
            {"source1_entity_id": "S1-C", "candidate_entity_id": "S2-4", "label": 0},
        ]
    )
    train_df, val_df = entity_aware_train_val_split(df, val_size=0.33, random_state=42)
    s1_train = set(train_df["source1_entity_id"])
    s1_val = set(val_df["source1_entity_id"])
    assert len(s1_train.intersection(s1_val)) == 0


def test_train_and_predict_smoke():
    X = np.array([[0.9, 0.8, 1.0], [0.1, 0.2, 0.0], [0.85, 0.9, 1.0], [0.2, 0.1, 0.0]])
    y = np.array([1, 0, 1, 0])
    model = train_matching_model(X, y, model_type="lightgbm", params={"n_estimators": 10, "min_child_samples": 1})
    probs = predict_match_probabilities(model, X)
    assert len(probs) == 4
    assert probs[0] > probs[1]
