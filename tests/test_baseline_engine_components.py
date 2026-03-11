"""Tests for baseline engine components built on shared training architecture."""

from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.infrastructure.checkpointing.store import JsonCheckpointStore
from eurosat_classifier.infrastructure.evaluation.baseline_evaluator import BaselineEvaluator
from eurosat_classifier.infrastructure.models.factory import SharedModelFactory
from eurosat_classifier.infrastructure.training.baseline_trainer import BaselineTrainer
from eurosat_classifier.infrastructure.training.split_json_loader import SplitJsonLoaderFactory


class BaselineEngineComponentsTests(unittest.TestCase):
    @staticmethod
    def _create_image(path: Path) -> None:
        image = Image.new("RGB", (64, 64), (64, 128, 192))
        image.save(path.as_posix(), format="JPEG")

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

            image_train = tmp_dir / "train.jpg"
            image_validation = tmp_dir / "validation.jpg"
            image_test = tmp_dir / "test.jpg"
            self._create_image(image_train)
            self._create_image(image_validation)
            self._create_image(image_test)

            train.write_text(
                json.dumps([{"path": image_train.as_posix(), "class_index": 0}]),
                encoding="utf-8",
            )
            validation.write_text(
                json.dumps([{"path": image_validation.as_posix(), "class_index": 1}]),
                encoding="utf-8",
            )
            test.write_text(
                json.dumps([{"path": image_test.as_posix(), "class_index": 2}]),
                encoding="utf-8",
            )

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
        evaluator = BaselineEvaluator([f"Class{i}" for i in range(10)])

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train_split = tmp_dir / "train_split.json"
            validation_split = tmp_dir / "validation_split.json"
            test_split = tmp_dir / "test_split.json"

            samples: list[dict[str, object]] = []
            for idx in range(6):
                image_path = tmp_dir / f"sample_{idx}.jpg"
                self._create_image(image_path)
                samples.append({"path": image_path.as_posix(), "class_index": idx % 3})

            train_split.write_text(json.dumps(samples[:4]), encoding="utf-8")
            validation_split.write_text(json.dumps(samples[4:5]), encoding="utf-8")
            test_split.write_text(json.dumps(samples[3:6]), encoding="utf-8")

            loaders = SplitJsonLoaderFactory().create(
                {
                    "train": train_split.as_posix(),
                    "validation": validation_split.as_posix(),
                    "test": test_split.as_posix(),
                },
                batch_size=2,
            )

            state = trainer.train(model, loaders, epochs=2, early_stopping_patience=1)
            summary = evaluator.evaluate(model, loaders["test"])
        finally:
            shutil.rmtree(tmp_dir)

        self.assertIn("epochs_ran", state)
        self.assertGreaterEqual(summary.accuracy, 0.0)
        self.assertIsNotNone(summary.confusion_matrix)
        self.assertIn("epoch_logs", state)
        self.assertIn("best_validation_f1", state)

    def test_checkpoint_metadata_contains_hyperparameters(self) -> None:
        model = SharedModelFactory().create("baseline_cnn")
        store = JsonCheckpointStore()

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            training_state = {
                "learning_rate": 0.001,
                "batch_size": 16,
                "epochs_requested": 5,
                "epochs_ran": 3,
                "early_stopping_patience": 2,
            }
            checkpoint_path = store.save_best(model, training_state, tmp_dir.as_posix())

            metadata_path = Path(checkpoint_path).with_suffix(".metadata.json")
            metadata_content = metadata_path.read_text(encoding="utf-8")

            self.assertIn('"hyperparameters"', metadata_content)
            self.assertIn('"learning_rate": 0.001', metadata_content)
            self.assertIn('"batch_size": 16', metadata_content)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
