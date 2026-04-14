import pandas as pd

from src.toxscreen.evaluation import recommend_operating_threshold


def test_threshold_policy_prefers_high_recall_above_precision_floor():
    threshold_table = pd.DataFrame(
        [
            {"threshold": 0.3, "precision": 0.18, "recall": 0.95, "f1": 0.30, "fn": 1, "fp": 40},
            {"threshold": 0.4, "precision": 0.24, "recall": 0.90, "f1": 0.38, "fn": 2, "fp": 24},
            {"threshold": 0.5, "precision": 0.31, "recall": 0.82, "f1": 0.45, "fn": 4, "fp": 15},
        ]
    )

    recommendation = recommend_operating_threshold(threshold_table, minimum_precision=0.20)

    assert recommendation["threshold"] == 0.4
    assert recommendation["false_negatives"] == 2
