from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import load_config
from .data import load_dataset, prepare_model_frame, profile_dataset
from .eda import build_eda_tables, save_eda_figures, save_eda_tables, write_eda_summary
from .evaluation import (
    build_threshold_table,
    compute_metrics,
    recommend_operating_threshold,
    save_confusion_matrix,
    save_threshold_plot,
)


METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")


def ensure_output_dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def write_results_summary(results: dict) -> None:
    """Write a plain-English summary of the run for fast review."""
    dataset_profile = results["dataset_profile"]
    comparison_path = "outputs/metrics/model_comparison.csv"
    threshold_path = "outputs/metrics/threshold_analysis.csv"

    best_model_name = max(
        results["models"].items(),
        key=lambda item: (item[1]["pr_auc"], item[1]["recall"], item[1]["f1"]),
    )[0]
    best_model = results["models"][best_model_name]

    threshold_policy = results.get("threshold_policy", {})
    recommended = threshold_policy.get("recommended_operating_point", {})

    lines = [
        "# Results Summary",
        "",
        "## Project story",
        "",
        "We use molecular structure data to predict clinical toxicity risk early, with special attention to false negatives.",
        "",
        "## Dataset snapshot",
        "",
        f"- Rows after basic filtering: {dataset_profile['rows_after_basic_filtering']}",
        f"- Toxic class is minority class: {dataset_profile['toxic_class_is_minority']}",
        f"- Class counts: {dataset_profile['class_counts']}",
        f"- Class percentages: {dataset_profile['class_percentages']}",
        "",
        "## Best held-out model",
        "",
        f"- Selected by PR-AUC, recall, and F1: `{best_model_name}`",
        f"- Precision: {best_model['precision']:.3f}",
        f"- Recall: {best_model['recall']:.3f}",
        f"- F1: {best_model['f1']:.3f}",
        f"- ROC-AUC: {best_model['roc_auc']:.3f}",
        f"- PR-AUC: {best_model['pr_auc']:.3f}",
        "",
    ]

    if recommended:
        lines.extend(
            [
                "## Safety-first threshold recommendation",
                "",
                f"- Optimized on: `{threshold_policy.get('optimized_on_model', 'logistic_regression')}`",
                f"- Minimum precision floor: {threshold_policy.get('minimum_precision_floor', 'n/a')}",
                f"- Recommended threshold: {recommended['threshold']:.2f}",
                f"- Precision at recommended threshold: {recommended['precision']:.3f}",
                f"- Recall at recommended threshold: {recommended['recall']:.3f}",
                f"- False negatives at recommended threshold: {recommended['false_negatives']}",
                f"- False positives at recommended threshold: {recommended['false_positives']}",
                "",
                "## Interpretation",
                "",
                "The central decision question is not only which model scores highest, but which operating threshold best reduces dangerous misses.",
                "A false negative means a toxic compound is predicted as safe. This repository treats that error as more costly than an extra false positive review burden.",
                f"For the full tradeoff table, see `{threshold_path}`.",
                f"For side-by-side model metrics, see `{comparison_path}`.",
            ]
        )

    with open(METRICS_DIR / "results_summary.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def run_audit(data_path: str | Path) -> dict:
    """Audit the dataset and persist the profile plus EDA artifacts."""
    ensure_output_dirs()
    df = load_dataset(data_path)
    profile = profile_dataset(df)
    eda_tables = build_eda_tables(df)

    save_eda_tables(eda_tables, METRICS_DIR)
    save_eda_figures(df, FIGURES_DIR)
    write_eda_summary(df, METRICS_DIR)

    with open(METRICS_DIR / "dataset_profile.json", "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    return profile


def run_training(data_path: str | Path, config_path: str | Path) -> dict:
    """Run the full training pipeline."""
    from .features import featurize_smiles
    from .modeling import build_models

    ensure_output_dirs()
    config = load_config(config_path)
    raw_df = load_dataset(data_path)
    dataset_profile = profile_dataset(raw_df)
    model_df = prepare_model_frame(raw_df)

    radius = config["feature_settings"]["fingerprint_radius"]
    n_bits = config["feature_settings"]["fingerprint_bits"]

    X, valid_indices = featurize_smiles(model_df["smiles"], radius=radius, n_bits=n_bits)
    valid_df = model_df.iloc[valid_indices].reset_index(drop=True)
    y = valid_df["ct_tox"].to_numpy()

    split_settings = config["train_settings"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_settings["test_size"],
        random_state=split_settings["random_state"],
        stratify=y,
    )

    models = build_models(config)
    thresholds = config["threshold_policy"]["candidate_thresholds"]
    minimum_precision = config["threshold_policy"]["minimum_precision"]

    all_metrics = {
        "dataset_profile": dataset_profile,
        "run_metadata": {
            "n_valid_molecules": int(len(valid_df)),
            "feature_dimensions": int(X.shape[1]),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
        },
        "models": {},
    }

    comparison_rows = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        all_metrics["models"][model_name] = metrics

        comparison_rows.append(
            {
                "model": model_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
            }
        )

        save_confusion_matrix(
            y_test,
            y_pred,
            FIGURES_DIR / f"confusion_matrix_{model_name}.png",
            title=f"Confusion Matrix: {model_name}",
        )

        if model_name == "logistic_regression":
            threshold_table = build_threshold_table(y_test, y_proba, thresholds)
            threshold_table.to_csv(METRICS_DIR / "threshold_analysis.csv", index=False)
            save_threshold_plot(threshold_table, FIGURES_DIR / "threshold_tradeoffs.png")

            all_metrics["threshold_policy"] = {
                "optimized_on_model": model_name,
                "minimum_precision_floor": minimum_precision,
                "recommended_operating_point": recommend_operating_threshold(
                    threshold_table,
                    minimum_precision=minimum_precision,
                ),
            }

    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        by=["pr_auc", "recall", "f1"],
        ascending=False,
    )
    comparison_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)

    with open(METRICS_DIR / "dataset_profile.json", "w", encoding="utf-8") as file:
        json.dump(dataset_profile, file, indent=2)

    with open(METRICS_DIR / "model_metrics.json", "w", encoding="utf-8") as file:
        json.dump(all_metrics, file, indent=2)

    write_results_summary(all_metrics)

    return all_metrics
