"""Tests for scaffold configuration loading."""

from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()
