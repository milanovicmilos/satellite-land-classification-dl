"""Tests for scaffold configuration loading."""

import json
from pathlib import Path
import tempfile
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.infrastructure.config_loader import JsonConfigLoader


class JsonConfigLoaderTests(unittest.TestCase):
    def test_load_returns_training_config(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        config = loader.load(str(PROJECT_ROOT / "configs" / "baseline.example.json"))

        self.assertEqual(config.experiment_name, "baseline-cnn-example")
        self.assertEqual(config.model_name, "baseline_cnn")
        self.assertEqual(config.split.seed, 42)

    def test_load_merges_minimal_config_with_defaults(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        config = loader.load(str(PROJECT_ROOT / "configs" / "baseline.minimal.json"))

        self.assertEqual(config.experiment_name, "baseline-cnn-minimal")
        self.assertEqual(config.dataset_root, "data/EuroSAT")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.split.seed, 42)

    def test_load_user_config_overrides_defaults_including_nested_dicts(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        override_payload = {
            "experiment_name": "override-check",
            "model": {"name": "baseline_cnn"},
            "training": {"batch_size": 8},
            "split": {"seed": 99},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(override_payload, handle)
            config_path = handle.name

        try:
            config = loader.load(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

        self.assertEqual(config.experiment_name, "override-check")
        self.assertEqual(config.batch_size, 8)
        self.assertEqual(config.split.seed, 99)
        self.assertEqual(config.epochs, 25)


if __name__ == "__main__":
    unittest.main()
