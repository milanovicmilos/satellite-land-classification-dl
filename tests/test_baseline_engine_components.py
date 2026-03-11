"""Tests for baseline engine components built on shared training architecture."""

from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.infrastructure.evaluation.baseline_evaluator import BaselineEvaluator
from eurosat_classifier.infrastructure.models.factory import SharedModelFactory
from eurosat_classifier.infrastructure.training.baseline_trainer import BaselineTrainer
from eurosat_classifier.infrastructure.training.split_json_loader import SplitJsonLoaderFactory


class BaselineEngineComponentsTests(unittest.TestCase):
    def test_model_factory_creates_baseline_model(self) -> None:
        factory = SharedModelFactory()

        model = factory.create("baseline_cnn")

        self.assertEqual(model.num_classes, 10)

    def test_split_loader_reads_json_files(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train = tmp_dir / "train_split.json"
            validation = tmp_dir / "validation_split.json"
            test = tmp_dir / "test_split.json"

            train.write_text(json.dumps([{"class_index": 0}]), encoding="utf-8")
            validation.write_text(json.dumps([{"class_index": 1}]), encoding="utf-8")
            test.write_text(json.dumps([{"class_index": 2}]), encoding="utf-8")

            loader_factory = SplitJsonLoaderFactory()
            loaders = loader_factory.create(
                {
                    "train": train.as_posix(),
                    "validation": validation.as_posix(),
                    "test": test.as_posix(),
                },
                batch_size=16,
            )

            self.assertEqual(len(loaders["train"]), 1)
            self.assertEqual(len(loaders["validation"]), 1)
            self.assertEqual(len(loaders["test"]), 1)
        finally:
            shutil.rmtree(tmp_dir)

    def test_trainer_and_evaluator_produce_metrics(self) -> None:
        model = SharedModelFactory().create("baseline_cnn")
        trainer = BaselineTrainer()
        evaluator = BaselineEvaluator(["A", "B", "C"])

        loaders = {
            "train": [{"class_index": 1}, {"class_index": 1}, {"class_index": 0}],
            "validation": [{"class_index": 1}],
            "test": [{"class_index": 1}, {"class_index": 0}, {"class_index": 2}],
        }

        state = trainer.train(model, loaders, epochs=10, early_stopping_patience=2)
        summary = evaluator.evaluate(model, loaders["test"])

        self.assertIn("epochs_ran", state)
        self.assertGreaterEqual(summary.accuracy, 0.0)
        self.assertIsNotNone(summary.confusion_matrix)


if __name__ == "__main__":
    unittest.main()
