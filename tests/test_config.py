from pathlib import Path

from src.toxscreen.config import load_config


def test_load_config_reads_defaults():
    config = load_config(Path("configs/defaults.json"))

    assert config["feature_settings"]["fingerprint_radius"] == 2
    assert config["feature_settings"]["fingerprint_bits"] == 2048
    assert "logistic_regression" in config["models"]
    assert "random_forest" in config["models"]
