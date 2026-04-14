from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return the project metric suite."""
    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def build_threshold_table(y_true, y_proba, thresholds: list[float]) -> pd.DataFrame:
    """Measure model behavior across operating thresholds."""
    rows = []
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        rows.append(
            {
                "threshold": float(threshold),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
                "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def recommend_operating_threshold(
    threshold_table: pd.DataFrame,
    minimum_precision: float,
) -> dict:
    """
    Pick an operating threshold with a safety-first bias.

    Preference order:
    1. Highest recall above a minimum precision floor
    2. Highest F1 as a tiebreaker
    3. Lowest threshold as a final tiebreaker
    """
    candidates = threshold_table[threshold_table["precision"] >= minimum_precision]
    if candidates.empty:
        candidates = threshold_table.copy()

    best = candidates.sort_values(
        by=["recall", "f1", "threshold"],
        ascending=[False, False, True],
    ).iloc[0]

    return {
        "threshold": float(best["threshold"]),
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "false_negatives": int(best["fn"]),
        "false_positives": int(best["fp"]),
    }


def save_confusion_matrix(y_true, y_pred, output_path: Path, title: str) -> None:
    """Persist a confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_threshold_plot(threshold_table: pd.DataFrame, output_path: Path) -> None:
    """Persist the threshold tradeoff figure."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold_table["threshold"], threshold_table["recall"], linewidth=2, label="Recall")
    ax.plot(
        threshold_table["threshold"],
        threshold_table["precision"],
        linewidth=2,
        label="Precision",
    )
    ax.plot(
        threshold_table["threshold"],
        threshold_table["false_negative_rate"],
        linewidth=2,
        label="False Negative Rate",
    )
    ax.plot(
        threshold_table["threshold"],
        threshold_table["false_positive_rate"],
        linewidth=2,
        label="False Positive Rate",
    )
    ax.set_title("Threshold Tradeoffs for Toxicity Triage")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
