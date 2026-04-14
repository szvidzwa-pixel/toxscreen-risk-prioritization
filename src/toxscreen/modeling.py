from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate


def build_models(config: dict) -> dict:
    """Instantiate all configured models."""
    return {
        "logistic_regression": LogisticRegression(**config["models"]["logistic_regression"]),
        "random_forest": RandomForestClassifier(**config["models"]["random_forest"]),
    }


def build_cv(config: dict) -> StratifiedKFold:
    """Create the stratified cross-validation splitter."""
    cv_settings = config["cv_settings"]
    return StratifiedKFold(
        n_splits=cv_settings["n_splits"],
        shuffle=cv_settings["shuffle"],
        random_state=cv_settings["random_state"],
    )


def tune_model(model_name: str, model, X_train, y_train, config: dict):
    """Run light hyperparameter tuning on the training split only."""
    grid = config["tuning"].get(model_name, {})
    if not grid:
        return model, {"best_params": {}, "best_score": None}

    search = GridSearchCV(
        estimator=model,
        param_grid=grid,
        scoring=config["tuning"]["primary_scoring"],
        cv=build_cv(config),
        n_jobs=1,
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
    }


def cross_validate_model(model, X_train, y_train, config: dict) -> dict:
    """Estimate model stability using cross-validation on the training split."""
    scores = cross_validate(
        clone(model),
        X_train,
        y_train,
        cv=build_cv(config),
        scoring={
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
        },
        n_jobs=1,
    )

    summary = {}
    for metric_name, values in scores.items():
        if not metric_name.startswith("test_"):
            continue
        clean_name = metric_name.replace("test_", "")
        summary[clean_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    return summary
