from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_models(config: dict) -> dict:
    """Instantiate all configured models."""
    return {
        "logistic_regression": LogisticRegression(**config["models"]["logistic_regression"]),
        "random_forest": RandomForestClassifier(**config["models"]["random_forest"]),
    }
