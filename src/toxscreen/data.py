from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"smiles", "ct_tox"}


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load ClinTox-style toxicity data and normalize column names."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df.columns = [str(column).strip().lower() for column in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """Create a business-friendly summary of the dataset."""
    profiled = df.dropna(subset=["smiles", "ct_tox"]).copy()
    profiled["ct_tox"] = profiled["ct_tox"].astype(int)

    counts = profiled["ct_tox"].value_counts().sort_index().to_dict()
    total = int(len(profiled))

    percentages = {
        str(label): round((count / total) * 100, 2) if total else 0.0
        for label, count in counts.items()
    }

    return {
        "rows_total": int(len(df)),
        "rows_after_basic_filtering": total,
        "missing_smiles": int(df["smiles"].isna().sum()),
        "missing_ct_tox": int(df["ct_tox"].isna().sum()),
        "class_counts": {str(label): int(count) for label, count in counts.items()},
        "class_percentages": percentages,
        "toxic_class_is_minority": bool(counts.get(1, 0) < counts.get(0, 0)),
    }


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a minimal frame ready for molecular featurization."""
    model_df = df.dropna(subset=["smiles", "ct_tox"]).copy()
    model_df["ct_tox"] = model_df["ct_tox"].astype(int)
    return model_df[["smiles", "ct_tox"]].reset_index(drop=True)
