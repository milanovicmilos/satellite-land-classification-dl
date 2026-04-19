"""Smoke tests for CLI orchestration and short baseline training path."""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.entrypoints.cli import main
from eurosat_classifier.infrastructure.datasets.eurosat_index import EXPECTED_EUROSAT_CLASSES


class TrainingSmokeTests(unittest.TestCase):
    @staticmethod
    def _create_image(path: Path) -> None:
        image = Image.new("RGB", (64, 64), (120, 160, 200))
        image.save(path.as_posix(), format="JPEG")

    def _create_dataset(self, dataset_root: Path, samples_per_class: int = 5) -> None:
        for class_name in EXPECTED_EUROSAT_CLASSES:
            class_dir = dataset_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for index in range(samples_per_class):
                self._create_image(class_dir / f"{class_name}_{index}.jpg")

    @staticmethod
    def _write_config(config_path: Path, dataset_root: Path) -> None:
        payload = {
            "experiment_name": "baseline-smoke",
            "dataset_root": dataset_root.as_posix(),
            "split": {
                "train_ratio": 0.6,
                "validation_ratio": 0.2,
                "test_ratio": 0.2,
                "seed": 42,
                "stratified": True,
            },
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "early_stopping_patience": 1,
                "early_stopping_min_delta": 0.0,
                "learning_rate": 0.001,
                "scheduler_factor": 0.5,
                "scheduler_patience": 0,
                "min_learning_rate": 0.000001,
                "augmentation_mode": "none",
            },
            "model": {
                "name": "baseline_cnn",
            },
        }
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run_cli(self, args: list[str]) -> int:
        with patch.object(sys, "argv", ["smoke-cli", *args]):
            with patch("sys.stdout", new=io.StringIO()):
                return main()

    def test_prepare_dataset_cli_smoke(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            dataset_root = tmp_dir / "EuroSAT"
            self._create_dataset(dataset_root)

            config_path = tmp_dir / "smoke_config.json"
            self._write_config(config_path, dataset_root)

            splits_dir = tmp_dir / "splits"
            exit_code = self._run_cli(
                [
                    "--prepare-dataset",
                    "--config",
                    config_path.as_posix(),
                    "--defaults",
                    (PROJECT_ROOT / "configs" / "experiment.defaults.json").as_posix(),
                    "--splits-output",
                    splits_dir.as_posix(),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((splits_dir / "train_split.json").exists())
            self.assertTrue((splits_dir / "validation_split.json").exists())
            self.assertTrue((splits_dir / "test_split.json").exists())
            self.assertTrue((splits_dir / "split_manifest.json").exists())
        finally:
            shutil.rmtree(tmp_dir)

    def test_run_baseline_cli_smoke(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            dataset_root = tmp_dir / "EuroSAT"
            self._create_dataset(dataset_root)

            config_path = tmp_dir / "smoke_config.json"
            self._write_config(config_path, dataset_root)

            defaults_path = (PROJECT_ROOT / "configs" / "experiment.defaults.json").as_posix()
            splits_dir = tmp_dir / "splits"
            reports_path = tmp_dir / "reports" / "baseline_smoke.json"
            checkpoints_dir = tmp_dir / "checkpoints"

            prepare_exit_code = self._run_cli(
                [
                    "--prepare-dataset",
                    "--config",
                    config_path.as_posix(),
                    "--defaults",
                    defaults_path,
                    "--splits-output",
                    splits_dir.as_posix(),
                ]
            )
            self.assertEqual(prepare_exit_code, 0)

            run_exit_code = self._run_cli(
                [
                    "--run-baseline",
                    "--config",
                    config_path.as_posix(),
                    "--defaults",
                    defaults_path,
                    "--splits-output",
                    splits_dir.as_posix(),
                    "--reports-output",
                    reports_path.as_posix(),
                    "--checkpoints-output",
                    checkpoints_dir.as_posix(),
                ]
            )

            self.assertEqual(run_exit_code, 0)
            self.assertTrue(reports_path.exists())
            self.assertTrue((checkpoints_dir / "best_checkpoint.pt").exists())
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
