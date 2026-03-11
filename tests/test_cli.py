"""Tests for CLI-facing orchestration code."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.entrypoints.cli import DryRunTrainingRunner, build_parser
from eurosat_classifier.domain.entities import DatasetSplit, Experiment


class CliTests(unittest.TestCase):
    def test_build_parser_supports_dry_run(self) -> None:
        parser = build_parser()

        parsed = parser.parse_args(["--dry-run"])

        self.assertTrue(parsed.dry_run)
        self.assertEqual(parsed.defaults, "configs/experiment.defaults.json")

    def test_dry_run_training_runner_returns_json_payload(self) -> None:
        runner = DryRunTrainingRunner()
        experiment = Experiment(
            name="demo",
            dataset_root="data/EuroSAT",
            model_name="baseline_cnn",
            split=DatasetSplit(0.7, 0.15, 0.15, 42),
        )

        payload = runner.run(experiment)

        self.assertIn('"experiment_name": "demo"', payload)
        self.assertIn('"model_name": "baseline_cnn"', payload)


if __name__ == "__main__":
    unittest.main()
