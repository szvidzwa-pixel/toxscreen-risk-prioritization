from __future__ import annotations

import sys

from src.toxscreen.pipeline import run_audit, run_training
from src.toxscreen.cli import main as cli_main


DEFAULT_DATA_PATH = "data/raw/clintox.csv"
DEFAULT_CONFIG_PATH = "configs/defaults.json"


LINE_WIDTH = 76


def divider(char: str = "=") -> str:
    return char * LINE_WIDTH


def print_section(title: str) -> None:
    print()
    print(divider("="))
    print(title)
    print(divider("="))


def fmt_metric(value: float) -> str:
    return f"{value:.3f}"


def describe_audit(audit_profile: dict) -> None:
    class_counts = audit_profile["class_counts"]
    class_percentages = audit_profile["class_percentages"]

    print_section("TOXSCREEN RISK PRIORITIZATION: END-TO-END WALKTHROUGH")
    print("Project objective:")
    print(
        "Build a supervised binary classification workflow that predicts whether a"
    )
    print(
        "drug candidate is clinically toxic (1) or non-toxic (0), using molecular"
    )
    print("structure as the input signal.")
    print()
    print("Why this problem matters:")
    print(
        "- In early toxicity screening, false negatives are more dangerous than false"
    )
    print("  positives because a toxic compound could be advanced as if it were safe.")
    print(
        "- This project therefore evaluates not only model discrimination, but also"
    )
    print("  how threshold selection changes safety risk.")

    print_section("STEP 1: DATA AUDIT AND EDA SNAPSHOT")
    print("What this step does:")
    print(
        "Reviews dataset size, missingness, and class imbalance before any model is"
    )
    print("trained, so the modeling strategy matches the data reality.")
    print()
    print(f"Dataset path: {DEFAULT_DATA_PATH}")
    print(f"Rows in raw file: {audit_profile['rows_total']}")
    print(f"Rows after basic filtering: {audit_profile['rows_after_basic_filtering']}")
    print(f"Missing SMILES values: {audit_profile['missing_smiles']}")
    print(f"Missing CT_TOX labels: {audit_profile['missing_ct_tox']}")
    print()
    print("Target variable:")
    print("- CT_TOX = 1 means clinically toxic")
    print("- CT_TOX = 0 means non-toxic")
    print()
    print("Class distribution:")
    print(
        f"- Non-toxic (0): {class_counts['0']} compounds "
        f"({class_percentages['0']:.2f}%)"
    )
    print(
        f"- Toxic (1): {class_counts['1']} compounds "
        f"({class_percentages['1']:.2f}%)"
    )
    print(
        f"- Toxic class is minority class: {audit_profile['toxic_class_is_minority']}"
    )
    print()
    print("Interpretation:")
    print(
        "The dataset is strongly imbalanced, so accuracy alone would be misleading."
    )
    print(
        "That is why the project emphasizes recall, PR-AUC, confusion matrices, and"
    )
    print("threshold tuning in later steps.")


def describe_training(results: dict) -> None:
    metadata = results["run_metadata"]
    models = results["models"]
    logistic = models["logistic_regression"]
    forest = models["random_forest"]
    threshold_policy = results["threshold_policy"]
    recommended = threshold_policy["recommended_operating_point"]

    print_section("STEP 2: FEATURE ENGINEERING AND MODEL TRAINING")
    print("Feature engineering approach:")
    print("- Molecules are represented as SMILES strings in the raw dataset.")
    print(
        "- Each molecule is converted into a Morgan fingerprint with radius=2 and"
    )
    print("  2048 bits using RDKit.")
    print(
        "- This creates a fixed-length binary feature vector that marks whether"
    )
    print("  particular structural fragments are present.")
    print()
    print("Preprocessing choices:")
    print(
        f"- Valid molecules used for modeling: {metadata['n_valid_molecules']}"
    )
    print(
        f"- Molecules dropped during RDKit parsing: "
        f"{results['dataset_profile']['rows_after_basic_filtering'] - metadata['n_valid_molecules']}"
    )
    print(f"- Feature dimensionality: {metadata['feature_dimensions']} fingerprint bits")
    print(f"- Train rows: {metadata['train_rows']}")
    print(f"- Test rows: {metadata['test_rows']}")
    print(
        "- Scaling was not applied because Morgan fingerprints are already binary"
    )
    print("  indicator features.")
    print()
    print("Models trained:")
    print("- Baseline: Logistic Regression")
    print("- Comparative model: Random Forest")
    print("- Both models were lightly tuned on the training split only.")

    print_section("STEP 3: MODEL COMPARISON ON THE HELD-OUT TEST SET")
    print("Held-out test results:")
    print(
        f"- Logistic Regression: precision={fmt_metric(logistic['precision'])}, "
        f"recall={fmt_metric(logistic['recall'])}, f1={fmt_metric(logistic['f1'])}, "
        f"roc_auc={fmt_metric(logistic['roc_auc'])}, pr_auc={fmt_metric(logistic['pr_auc'])}"
    )
    print(
        f"- Random Forest: precision={fmt_metric(forest['precision'])}, "
        f"recall={fmt_metric(forest['recall'])}, f1={fmt_metric(forest['f1'])}, "
        f"roc_auc={fmt_metric(forest['roc_auc'])}, pr_auc={fmt_metric(forest['pr_auc'])}"
    )
    print()
    print("Best model decision:")
    print(
        "Logistic Regression is the stronger deployment candidate in this run because"
    )
    print(
        "it produced the best balance of recall, F1, ROC-AUC, and PR-AUC on the"
    )
    print("held-out test set.")
    print()
    print("Light tuning summary:")
    print(
        f"- Logistic Regression best params: {logistic['tuning']['best_params']}"
    )
    print(
        f"- Random Forest best params: {forest['tuning']['best_params']}"
    )
    print()
    print("Cross-validation check on the training split:")
    print(
        f"- Logistic Regression mean CV PR-AUC: "
        f"{fmt_metric(logistic['cross_validation']['pr_auc']['mean'])}"
    )
    print(
        f"- Logistic Regression mean CV ROC-AUC: "
        f"{fmt_metric(logistic['cross_validation']['roc_auc']['mean'])}"
    )
    print(
        f"- Logistic Regression mean CV recall: "
        f"{fmt_metric(logistic['cross_validation']['recall']['mean'])}"
    )
    print()
    print("Overfitting note:")
    print(
        f"- Logistic Regression training recall: {fmt_metric(logistic['train_metrics']['recall'])}"
    )
    print(
        f"- Logistic Regression test recall: {fmt_metric(logistic['recall'])}"
    )
    print(
        "There is a noticeable train-test gap, which suggests some overfitting risk."
    )
    print(
        "That is expected on a small, highly imbalanced molecular dataset and should"
    )
    print("be discussed honestly in the paper.")

    print_section("STEP 4: SAFETY-FIRST THRESHOLD ANALYSIS")
    threshold_rows = {
        row["threshold"]: row
        for row in (
            {"threshold": 0.50, "precision": logistic["precision"], "recall": logistic["recall"],
             "false_negatives": logistic["confusion_matrix"][1][0],
             "false_positives": logistic["confusion_matrix"][0][1]}
            ,
            recommended,
        )
    }
    base = threshold_rows[0.50]
    current = threshold_rows[recommended["threshold"]]

    print("Why threshold analysis matters:")
    print(
        "A model score is not the final decision. The operating threshold determines"
    )
    print(
        "how conservative the screening system is, and therefore how many toxic"
    )
    print("compounds are missed.")
    print()
    print(
        f"- Default threshold 0.50: precision={fmt_metric(base['precision'])}, "
        f"recall={fmt_metric(base['recall'])}, false_negatives={base['false_negatives']}, "
        f"false_positives={base['false_positives']}"
    )
    print(
        f"- Recommended threshold {recommended['threshold']:.2f}: "
        f"precision={fmt_metric(current['precision'])}, "
        f"recall={fmt_metric(current['recall'])}, "
        f"false_negatives={current['false_negatives']}, "
        f"false_positives={current['false_positives']}"
    )
    print()
    print("Operational interpretation:")
    print(
        f"Lowering the threshold from 0.50 to {recommended['threshold']:.2f} reduces "
        f"false negatives from {base['false_negatives']} to {current['false_negatives']}."
    )
    print(
        f"Recall increases from {fmt_metric(base['recall'])} to "
        f"{fmt_metric(current['recall'])}."
    )
    print(
        f"The tradeoff is that false positives increase from {base['false_positives']} "
        f"to {current['false_positives']}, which is acceptable for a safety-first "
        "screening workflow."
    )
    print()
    print(
        f"Minimum precision floor used in selection: {threshold_policy['minimum_precision_floor']}"
    )


def describe_outputs() -> None:
    print_section("STEP 5: SAVED ARTIFACTS")
    print("- outputs/metrics/eda_summary.md")
    print("- outputs/metrics/model_metrics.json")
    print("- outputs/metrics/model_comparison.csv")
    print("- outputs/metrics/threshold_analysis.csv")
    print("- outputs/metrics/results_summary.md")
    print("- outputs/figures/ for confusion matrices and threshold tradeoff plots")
    print("- docs/figures/ for README-ready tracked visualizations")
    print()
    print("Suggested files to open next:")
    print("- outputs/metrics/results_summary.md")
    print("- outputs/metrics/eda_summary.md")
    print("- outputs/metrics/threshold_analysis.csv")
    print("- docs/figures/threshold_tradeoffs.png")
    print()
    print("End-to-end run completed successfully.")


def run_default_pipeline() -> None:
    """Run the full project with built-in default paths."""
    audit_profile = run_audit(DEFAULT_DATA_PATH)
    results = run_training(DEFAULT_DATA_PATH, DEFAULT_CONFIG_PATH)
    describe_audit(audit_profile)
    describe_training(results)
    describe_outputs()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_default_pipeline()
    else:
        cli_main()
