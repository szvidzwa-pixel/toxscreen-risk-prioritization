from __future__ import annotations

import json
import sys

from src.toxscreen.pipeline import run_audit, run_training
from src.toxscreen.cli import main as cli_main


DEFAULT_DATA_PATH = "data/raw/clintox.csv"
DEFAULT_CONFIG_PATH = "configs/defaults.json"


def run_default_pipeline() -> None:
    """Run the full project with built-in default paths."""
    print("Running ToxScreen end-to-end with default project paths...")
    print(f"Dataset: {DEFAULT_DATA_PATH}")
    print(f"Config: {DEFAULT_CONFIG_PATH}")
    print()

    print("Step 1/2: auditing dataset")
    audit_profile = run_audit(DEFAULT_DATA_PATH)
    print(json.dumps(audit_profile, indent=2))
    print()

    print("Step 2/2: training models")
    results = run_training(DEFAULT_DATA_PATH, DEFAULT_CONFIG_PATH)

    print()
    print("End-to-end run completed.")
    print("Key outputs:")
    print("- outputs/metrics/eda_summary.md")
    print("- outputs/metrics/model_metrics.json")
    print("- outputs/metrics/results_summary.md")
    print("- outputs/figures/")
    print("- docs/figures/")
    print()
    print("Best held-out model metrics:")
    print(json.dumps(results["models"]["logistic_regression"], indent=2))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_default_pipeline()
    else:
        cli_main()
