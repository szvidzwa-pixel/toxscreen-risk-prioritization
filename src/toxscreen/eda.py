from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def build_eda_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the core EDA tables used in the written summary and plots."""
    working = df.copy()
    working["smiles_length"] = working["smiles"].astype(str).str.len()

    ct_counts = working["ct_tox"].value_counts(dropna=False).sort_index()
    ct_normalized = working["ct_tox"].value_counts(normalize=True, dropna=False).sort_index()

    fda_counts = None
    fda_normalized = None
    if "fda_approved" in working.columns:
        fda_counts = working["fda_approved"].value_counts(dropna=False).sort_index()
        fda_normalized = (
            working["fda_approved"].value_counts(normalize=True, dropna=False).sort_index()
        )

    tables = {
        "ct_tox_counts": ct_counts.rename_axis("ct_tox").reset_index(name="count"),
        "ct_tox_normalized": ct_normalized.rename_axis("ct_tox").reset_index(name="proportion"),
        "smiles_length_summary": working["smiles_length"].describe().reset_index(),
    }

    if fda_counts is not None and fda_normalized is not None:
        tables["fda_counts"] = fda_counts.rename_axis("fda_approved").reset_index(name="count")
        tables["fda_normalized"] = (
            fda_normalized.rename_axis("fda_approved").reset_index(name="proportion")
        )

    return tables


def save_eda_tables(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Save EDA tables to CSV for inspection and reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)


def save_eda_figures(df: pd.DataFrame, figure_dir: Path) -> None:
    """Save the main EDA visualizations."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    working = df.copy()
    working["smiles_length"] = working["smiles"].astype(str).str.len()

    ct_distribution = (
        working["ct_tox"].value_counts(normalize=True, dropna=False).sort_index() * 100
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Non-toxic (0)", "Toxic (1)"],
        ct_distribution.tolist(),
        color=["#7aa6c2", "#d95f5f"],
    )
    ax.set_title("ClinTox Target Distribution")
    ax.set_ylabel("Percent of compounds")
    ax.set_ylim(0, max(ct_distribution.max() * 1.15, 10))
    for idx, value in enumerate(ct_distribution.tolist()):
        ax.text(idx, value + 1, f"{value:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(figure_dir / "ct_tox_class_distribution.png", dpi=200)
    plt.close(fig)

    if "fda_approved" in working.columns:
        fda_distribution = (
            working["fda_approved"].value_counts(normalize=True, dropna=False).sort_index() * 100
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ["Not approved (0)", "Approved (1)"],
            fda_distribution.tolist(),
            color=["#c98f4a", "#63a375"],
        )
        ax.set_title("FDA Approval Label Distribution")
        ax.set_ylabel("Percent of compounds")
        ax.set_ylim(0, max(fda_distribution.max() * 1.15, 10))
        for idx, value in enumerate(fda_distribution.tolist()):
            ax.text(idx, value + 1, f"{value:.2f}%", ha="center")
        fig.tight_layout()
        fig.savefig(figure_dir / "fda_approved_distribution.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(working["smiles_length"], bins=30, color="#587291", edgecolor="white")
    ax.set_title("Distribution of SMILES String Lengths")
    ax.set_xlabel("SMILES length")
    ax.set_ylabel("Compound count")
    fig.tight_layout()
    fig.savefig(figure_dir / "smiles_length_distribution.png", dpi=200)
    plt.close(fig)


def write_eda_summary(df: pd.DataFrame, metrics_dir: Path) -> None:
    """Write a plain-English EDA summary for fast review."""
    tables = build_eda_tables(df)
    ct_counts = tables["ct_tox_counts"]
    ct_normalized = tables["ct_tox_normalized"]

    non_toxic_count = int(ct_counts.loc[ct_counts["ct_tox"] == 0, "count"].iloc[0])
    toxic_count = int(ct_counts.loc[ct_counts["ct_tox"] == 1, "count"].iloc[0])
    non_toxic_prop = float(
        ct_normalized.loc[ct_normalized["ct_tox"] == 0, "proportion"].iloc[0]
    )
    toxic_prop = float(ct_normalized.loc[ct_normalized["ct_tox"] == 1, "proportion"].iloc[0])
    smiles_length = df["smiles"].astype(str).str.len()

    lines = [
        "# EDA Summary",
        "",
        "## Core finding",
        "",
        "We use molecular structure data to predict clinical toxicity risk early, and the EDA confirms that the toxicity target is highly imbalanced.",
        "",
        "## Dataset overview",
        "",
        f"- Total rows: {len(df)}",
        f"- Columns: {list(df.columns)}",
        f"- Missing values: {df.isna().sum().to_dict()}",
        f"- Duplicate SMILES: {int(df['smiles'].duplicated().sum())}",
        "",
        "## Target distribution",
        "",
        f"- Non-toxic compounds (`CT_TOX = 0`): {non_toxic_count} ({non_toxic_prop:.2%})",
        f"- Toxic compounds (`CT_TOX = 1`): {toxic_count} ({toxic_prop:.2%})",
        "- Toxic compounds are the minority class, so accuracy alone would be misleading.",
        "- This class imbalance is why recall, PR-AUC, confusion matrices, and threshold tuning matter in the modeling stage.",
        "",
        "## Molecular structure proxy",
        "",
        f"- SMILES length mean: {smiles_length.mean():.2f}",
        f"- SMILES length median: {smiles_length.median():.2f}",
        f"- SMILES length min/max: {smiles_length.min()} / {smiles_length.max()}",
        "",
        "## Reproducible checks used in the EDA",
        "",
        "```python",
        "df['CT_TOX'].value_counts()",
        "df['CT_TOX'].value_counts(normalize=True)",
        "df['FDA_APPROVED'].value_counts(normalize=True)",
        "df['smiles'].str.len().describe()",
        "```",
    ]

    with open(metrics_dir / "eda_summary.md", "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")

    with open(metrics_dir / "eda_snapshot.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
                "duplicate_smiles": int(df["smiles"].duplicated().sum()),
                "ct_tox_distribution": {
                    "0": {"count": non_toxic_count, "proportion": round(non_toxic_prop, 6)},
                    "1": {"count": toxic_count, "proportion": round(toxic_prop, 6)},
                },
            },
            file,
            indent=2,
        )
