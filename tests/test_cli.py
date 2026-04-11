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
        self.assertEqual(parsed.config, "configs/baseline_cnn.json")
        self.assertEqual(parsed.defaults, "configs/experiment.defaults.json")
        self.assertEqual(parsed.splits_output, "artifacts/splits")
        self.assertIsNone(parsed.seed)

    def test_build_parser_supports_prepare_dataset(self) -> None:
        parser = build_parser()

        parsed = parser.parse_args(["--prepare-dataset"])

        self.assertTrue(parsed.prepare_dataset)

    def test_build_parser_supports_run_baseline(self) -> None:
        parser = build_parser()

        parsed = parser.parse_args(["--run-baseline"])

        self.assertTrue(parsed.run_baseline)
        self.assertEqual(parsed.reports_output, "artifacts/reports/baseline_metrics.json")

    def test_build_parser_supports_seed_override(self) -> None:
        parser = build_parser()

        parsed = parser.parse_args(["--dry-run", "--seed", "123"])

        self.assertEqual(parsed.seed, 123)

    def test_build_parser_supports_training_and_model_overrides(self) -> None:
        parser = build_parser()

        parsed = parser.parse_args(
            [
                "--run-baseline",
                "--model-name",
                "efficientnet_b0",
                "--epochs",
                "12",
                "--batch-size",
                "8",
                "--learning-rate",
                "0.0003",
                "--augmentation-mode",
                "flips",
                "--use-pretrained",
                "--freeze-backbone",
                "--no-stratified",
            ]
        )

        self.assertEqual(parsed.model_name, "efficientnet_b0")
        self.assertEqual(parsed.epochs, 12)
        self.assertEqual(parsed.batch_size, 8)
        self.assertAlmostEqual(parsed.learning_rate, 0.0003)
        self.assertEqual(parsed.augmentation_mode, "flips")
        self.assertTrue(parsed.use_pretrained)
        self.assertTrue(parsed.freeze_backbone)
        self.assertFalse(parsed.stratified)

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
