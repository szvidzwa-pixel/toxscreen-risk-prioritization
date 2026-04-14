from __future__ import annotations

import json
from pathlib import Path


def load_config(config_path: str | Path) -> dict:
    """Load a JSON config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
