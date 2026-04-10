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

        config = loader.load(str(PROJECT_ROOT / "configs" / "baseline_cnn_full.json"))

        self.assertEqual(config.experiment_name, "baseline-cnn-full")
        self.assertEqual(config.model_name, "baseline_cnn")
        self.assertEqual(config.model_options, {})
        self.assertEqual(config.split.seed, 42)

    def test_load_merges_minimal_config_with_defaults(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        payload = {
            "experiment_name": "baseline-cnn-minimal",
            "model": {"name": "baseline_cnn"},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            config = loader.load(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

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

    def test_load_reads_model_options_for_stage_config(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        config = loader.load(str(PROJECT_ROOT / "configs" / "efficientnet_b0.stage1.optimized.json"))

        self.assertEqual(config.model_name, "efficientnet_b0")
        self.assertTrue(bool(config.model_options.get("freeze_backbone")))
        self.assertTrue(bool(config.model_options.get("use_pretrained")))

    def test_load_reads_stage2_resume_from_path(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        config = loader.load(str(PROJECT_ROOT / "configs" / "efficientnet_b0.stage2.optimized.json"))

        self.assertEqual(config.model_name, "efficientnet_b0")
        self.assertFalse(bool(config.model_options.get("freeze_backbone")))
        self.assertEqual(
            config.resume_from,
            "checkpoints/efficientnet_b0/stage1/best_checkpoint.pt",
        )

    def test_load_reads_resnet_stage1_options(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        config = loader.load(str(PROJECT_ROOT / "configs" / "resnet50.stage1.optimized.json"))

        self.assertEqual(config.model_name, "resnet50")
        self.assertTrue(bool(config.model_options.get("freeze_backbone")))
        self.assertTrue(bool(config.model_options.get("use_pretrained")))

    def test_load_reads_resnet_stage2_resume_path(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        config = loader.load(str(PROJECT_ROOT / "configs" / "resnet50.stage2.optimized.json"))

        self.assertEqual(config.model_name, "resnet50")
        self.assertFalse(bool(config.model_options.get("freeze_backbone")))
        self.assertEqual(config.resume_from, "checkpoints/resnet50/stage1/best_checkpoint.pt")

    def test_load_sets_full_augmentation_for_efficientnet_when_omitted(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        payload = {
            "experiment_name": "eff-omitted-augmentation",
            "model": {"name": "efficientnet_b0"},
            "training": {"augmentation_mode": None},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            config = loader.load(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

        self.assertEqual(config.augmentation_mode, "full")

    def test_load_keeps_explicit_augmentation_mode_for_efficientnet(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        payload = {
            "experiment_name": "eff-explicit-augmentation",
            "model": {"name": "efficientnet_b0"},
            "training": {"augmentation_mode": "flips"},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            config = loader.load(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

        self.assertEqual(config.augmentation_mode, "flips")

    def test_load_keeps_null_augmentation_for_non_efficientnet_when_omitted(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        payload = {
            "experiment_name": "resnet-omitted-augmentation",
            "model": {"name": "resnet50"},
            "training": {"augmentation_mode": None},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            config = loader.load(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

        self.assertIsNone(config.augmentation_mode)

    def test_load_derives_scheduler_patience_from_early_stopping(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        payload = {
            "experiment_name": "scheduler-derived",
            "model": {"name": "baseline_cnn"},
            "training": {
                "early_stopping_patience": 5,
                "scheduler_patience": None,
            },
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            config = loader.load(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

        self.assertEqual(config.scheduler_patience, 2)

    def test_load_defaults_empty_model_options_when_missing(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        )

        payload = {
            "experiment_name": "no-model-options",
            "model": {"name": "baseline_cnn"},
            "training": {
                "epochs": 1,
                "batch_size": 4,
                "early_stopping_patience": 1,
            },
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            config = loader.load(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

        self.assertEqual(config.model_options, {})

    def test_load_applies_constructor_seed_override(self) -> None:
        loader = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json"),
            split_seed_override=123,
        )

        payload = {
            "experiment_name": "override-seed-test",
            "model": {"name": "efficientnet_b0"},
            "split": {"seed": 999},
        }

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            config_path = handle.name

        try:
            config = loader.load(config_path)
        finally:
            Path(config_path).unlink(missing_ok=True)

        self.assertEqual(config.split.seed, 123)


if __name__ == "__main__":
    unittest.main()
