"""Tests for baseline engine components built on shared training architecture."""

from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.infrastructure.checkpointing.store import JsonCheckpointStore
from eurosat_classifier.infrastructure.config_loader import JsonConfigLoader
from eurosat_classifier.infrastructure.evaluation.baseline_evaluator import BaselineEvaluator
from eurosat_classifier.infrastructure.evaluation.report_writer import JsonReportWriter
from eurosat_classifier.application.services.training_orchestrator import TrainingOrchestrator
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

    def test_model_factory_creates_efficientnet_model(self) -> None:
        factory = SharedModelFactory()

        model = factory.create("efficientnet_b0")

        self.assertEqual(model.num_classes, 10)

    def test_model_factory_creates_resnet50_model(self) -> None:
        factory = SharedModelFactory()

        model = factory.create("resnet50")

        self.assertEqual(model.num_classes, 10)

    def test_efficientnet_forward_shape_and_backbone_freeze_controls(self) -> None:
        model = SharedModelFactory().create("efficientnet_b0")

        model.set_backbone_trainable(False)
        self.assertTrue(model.backbone_frozen)
        self.assertTrue(all(not p.requires_grad for p in model.backbone.features.parameters()))

        model.set_backbone_trainable(True)
        self.assertFalse(model.backbone_frozen)
        self.assertTrue(all(p.requires_grad for p in model.backbone.features.parameters()))

        inputs = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 10))

    def test_resnet50_forward_shape_and_backbone_freeze_controls(self) -> None:
        model = SharedModelFactory().create(
            "resnet50",
            {"use_pretrained": False, "freeze_backbone": False},
        )

        model.set_backbone_trainable(False)
        self.assertTrue(model.backbone_frozen)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for name, parameter in model.backbone.named_parameters()
                if not name.startswith("fc.")
            )
        )
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.fc.parameters()))

        model.set_backbone_trainable(True)
        self.assertFalse(model.backbone_frozen)
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))

        inputs = torch.zeros((2, 3, 64, 64), dtype=torch.float32)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 10))

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

    def test_split_loader_applies_imagenet_normalization_only_for_efficientnet(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train = tmp_dir / "train_split.json"
            validation = tmp_dir / "validation_split.json"
            test = tmp_dir / "test_split.json"

            image_path = tmp_dir / "sample.jpg"
            image = Image.new("RGB", (64, 64), (255, 255, 255))
            image.save(image_path.as_posix(), format="JPEG")

            payload = json.dumps([{"path": image_path.as_posix(), "class_index": 0}])
            train.write_text(payload, encoding="utf-8")
            validation.write_text(payload, encoding="utf-8")
            test.write_text(payload, encoding="utf-8")

            split_paths = {
                "train": train.as_posix(),
                "validation": validation.as_posix(),
                "test": test.as_posix(),
            }

            baseline_loaders = SplitJsonLoaderFactory().create(
                split_paths,
                batch_size=1,
                model_name="baseline_cnn",
            )
            efficientnet_loaders = SplitJsonLoaderFactory().create(
                split_paths,
                batch_size=1,
                model_name="efficientnet_b0",
            )

            baseline_inputs, _ = next(iter(baseline_loaders["train"]))
            efficientnet_inputs, _ = next(iter(efficientnet_loaders["train"]))

            self.assertAlmostEqual(float(baseline_inputs[0, 0, 0, 0]), 1.0, places=3)
            self.assertGreater(float(efficientnet_inputs[0, 0, 0, 0]), 2.0)
        finally:
            shutil.rmtree(tmp_dir)

    def test_split_loader_supports_flips_only_augmentation_mode(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train = tmp_dir / "train_split.json"
            validation = tmp_dir / "validation_split.json"
            test = tmp_dir / "test_split.json"

            image_path = tmp_dir / "sample.jpg"
            image = Image.new("RGB", (64, 64), (255, 255, 255))
            image.save(image_path.as_posix(), format="JPEG")

            payload = json.dumps([{"path": image_path.as_posix(), "class_index": 0}])
            train.write_text(payload, encoding="utf-8")
            validation.write_text(payload, encoding="utf-8")
            test.write_text(payload, encoding="utf-8")

            split_paths = {
                "train": train.as_posix(),
                "validation": validation.as_posix(),
                "test": test.as_posix(),
            }

            loaders = SplitJsonLoaderFactory().create(
                split_paths,
                batch_size=1,
                model_name="efficientnet_b0",
                augmentation_mode="flips",
            )

            train_transforms = loaders["train"].dataset._transform.transforms
            transform_names = [type(transform).__name__ for transform in train_transforms]

            self.assertIn("RandomHorizontalFlip", transform_names)
            self.assertIn("RandomVerticalFlip", transform_names)
            self.assertIn("RandomChoice", transform_names)
            self.assertNotIn("RandomAffine", transform_names)

            right_angle_rotation = next(
                transform
                for transform in train_transforms
                if type(transform).__name__ == "RandomChoice"
            )
            rotation_names = [
                type(transform).__name__ for transform in right_angle_rotation.transforms
            ]
            self.assertEqual(rotation_names, ["RandomRotation"] * 4)

            rotation_degrees = {
                int(transform.degrees[0]) for transform in right_angle_rotation.transforms
            }
            self.assertEqual(rotation_degrees, {0, 90, 180, 270})
        finally:
            shutil.rmtree(tmp_dir)

    def test_split_loader_full_augmentation_keeps_existing_policy(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train = tmp_dir / "train_split.json"
            validation = tmp_dir / "validation_split.json"
            test = tmp_dir / "test_split.json"

            image_path = tmp_dir / "sample.jpg"
            image = Image.new("RGB", (64, 64), (255, 255, 255))
            image.save(image_path.as_posix(), format="JPEG")

            payload = json.dumps([{"path": image_path.as_posix(), "class_index": 0}])
            train.write_text(payload, encoding="utf-8")
            validation.write_text(payload, encoding="utf-8")
            test.write_text(payload, encoding="utf-8")

            split_paths = {
                "train": train.as_posix(),
                "validation": validation.as_posix(),
                "test": test.as_posix(),
            }

            loaders = SplitJsonLoaderFactory().create(
                split_paths,
                batch_size=1,
                model_name="efficientnet_b0",
                augmentation_mode="full",
            )

            train_transforms = loaders["train"].dataset._transform.transforms
            transform_names = [type(transform).__name__ for transform in train_transforms]

            self.assertIn("RandomHorizontalFlip", transform_names)
            self.assertIn("RandomVerticalFlip", transform_names)
            self.assertIn("RandomRotation", transform_names)
            self.assertIn("RandomAffine", transform_names)
            self.assertNotIn("RandomChoice", transform_names)
        finally:
            shutil.rmtree(tmp_dir)

    def test_split_loader_uses_registry_normalization_for_future_models(self) -> None:
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train = tmp_dir / "train_split.json"
            validation = tmp_dir / "validation_split.json"
            test = tmp_dir / "test_split.json"

            image_path = tmp_dir / "sample.jpg"
            image = Image.new("RGB", (64, 64), (255, 255, 255))
            image.save(image_path.as_posix(), format="JPEG")

            payload = json.dumps([{"path": image_path.as_posix(), "class_index": 0}])
            train.write_text(payload, encoding="utf-8")
            validation.write_text(payload, encoding="utf-8")
            test.write_text(payload, encoding="utf-8")

            split_paths = {
                "train": train.as_posix(),
                "validation": validation.as_posix(),
                "test": test.as_posix(),
            }

            resnet_loaders = SplitJsonLoaderFactory().create(
                split_paths,
                batch_size=1,
                model_name="resnet50",
            )

            resnet_inputs, _ = next(iter(resnet_loaders["train"]))
            self.assertGreater(float(resnet_inputs[0, 0, 0, 0]), 2.0)

            train_transforms = resnet_loaders["train"].dataset._transform.transforms
            transform_names = [type(transform).__name__ for transform in train_transforms]
            self.assertNotIn("RandomHorizontalFlip", transform_names)
            self.assertNotIn("RandomVerticalFlip", transform_names)
            self.assertNotIn("RandomRotation", transform_names)
            self.assertNotIn("RandomAffine", transform_names)
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

            state = trainer.train(
                model,
                loaders,
                epochs=2,
                early_stopping_patience=1,
                learning_rate=0.001,
                scheduler_factor=0.5,
                scheduler_patience=1,
                min_learning_rate=1e-6,
                early_stopping_min_delta=0.0,
            )
            summary = evaluator.evaluate(model, loaders["test"])
        finally:
            shutil.rmtree(tmp_dir)

        self.assertIn("epochs_ran", state)
        self.assertGreaterEqual(summary.accuracy, 0.0)
        self.assertIsNotNone(summary.confusion_matrix)
        self.assertIn("epoch_logs", state)
        self.assertIn("best_validation_f1", state)

    def test_trainer_preserves_zero_scheduler_patience(self) -> None:
        model = SharedModelFactory().create("baseline_cnn")
        trainer = BaselineTrainer()

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train_split = tmp_dir / "train_split.json"
            validation_split = tmp_dir / "validation_split.json"
            test_split = tmp_dir / "test_split.json"

            samples: list[dict[str, object]] = []
            for idx in range(6):
                image_path = tmp_dir / f"zero_scheduler_{idx}.jpg"
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

            state = trainer.train(
                model,
                loaders,
                epochs=2,
                early_stopping_patience=1,
                learning_rate=0.001,
                scheduler_factor=0.5,
                scheduler_patience=0,
                min_learning_rate=1e-6,
                early_stopping_min_delta=0.0,
            )
        finally:
            shutil.rmtree(tmp_dir)

        self.assertEqual(state["scheduler_patience"], 0)

    def test_trainer_raises_for_non_positive_learning_rate(self) -> None:
        model = SharedModelFactory().create("baseline_cnn")
        trainer = BaselineTrainer()

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train_split = tmp_dir / "train_split.json"
            validation_split = tmp_dir / "validation_split.json"
            test_split = tmp_dir / "test_split.json"

            samples: list[dict[str, object]] = []
            for idx in range(6):
                image_path = tmp_dir / f"lr_validation_{idx}.jpg"
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

            with self.assertRaises(ValueError):
                trainer.train(
                    model,
                    loaders,
                    epochs=2,
                    early_stopping_patience=1,
                    learning_rate=0.0,
                    scheduler_factor=0.5,
                    scheduler_patience=0,
                    min_learning_rate=1e-6,
                    early_stopping_min_delta=0.0,
                )
        finally:
            shutil.rmtree(tmp_dir)

    def test_trainer_raises_for_negative_scheduler_patience(self) -> None:
        model = SharedModelFactory().create("baseline_cnn")
        trainer = BaselineTrainer()

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train_split = tmp_dir / "train_split.json"
            validation_split = tmp_dir / "validation_split.json"
            test_split = tmp_dir / "test_split.json"

            samples: list[dict[str, object]] = []
            for idx in range(6):
                image_path = tmp_dir / f"scheduler_validation_{idx}.jpg"
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

            with self.assertRaises(ValueError):
                trainer.train(
                    model,
                    loaders,
                    epochs=2,
                    early_stopping_patience=1,
                    learning_rate=0.001,
                    scheduler_factor=0.5,
                    scheduler_patience=-1,
                    min_learning_rate=1e-6,
                    early_stopping_min_delta=0.0,
                )
        finally:
            shutil.rmtree(tmp_dir)

    def test_trainer_raises_for_non_positive_min_learning_rate(self) -> None:
        model = SharedModelFactory().create("baseline_cnn")
        trainer = BaselineTrainer()

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train_split = tmp_dir / "train_split.json"
            validation_split = tmp_dir / "validation_split.json"
            test_split = tmp_dir / "test_split.json"

            samples: list[dict[str, object]] = []
            for idx in range(6):
                image_path = tmp_dir / f"min_lr_validation_{idx}.jpg"
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

            with self.assertRaises(ValueError):
                trainer.train(
                    model,
                    loaders,
                    epochs=2,
                    early_stopping_patience=1,
                    learning_rate=0.001,
                    scheduler_factor=0.5,
                    scheduler_patience=0,
                    min_learning_rate=0.0,
                    early_stopping_min_delta=0.0,
                )
        finally:
            shutil.rmtree(tmp_dir)

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

    def test_load_checkpoint_raises_clear_error_for_missing_file(self) -> None:
        model = SharedModelFactory().create("baseline_cnn")
        store = JsonCheckpointStore()

        with self.assertRaises(FileNotFoundError) as context:
            store.load_checkpoint(model, "checkpoints/nonexistent/best_checkpoint.pt")

        self.assertIn("Checkpoint file not found", str(context.exception))

    def test_load_checkpoint_raises_clear_error_for_incompatible_architecture(self) -> None:
        store = JsonCheckpointStore()
        source_model = SharedModelFactory().create("baseline_cnn")
        target_model = SharedModelFactory().create("efficientnet_b0")

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            checkpoint_path = tmp_dir / "incompatible_checkpoint.pt"
            torch.save(source_model.state_dict(), checkpoint_path.as_posix())

            with self.assertRaises(RuntimeError) as context:
                store.load_checkpoint(target_model, checkpoint_path.as_posix())

            self.assertIn("Checkpoint incompatibility", str(context.exception))
        finally:
            shutil.rmtree(tmp_dir)

    def test_efficientnet_stage1_smoke_runs_one_epoch(self) -> None:
        config = JsonConfigLoader(
            defaults_path=str(PROJECT_ROOT / "configs" / "experiment.defaults.json")
        ).load(str(PROJECT_ROOT / "configs" / "efficientnet_b0.stage1.optimized.json"))

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            train_split = tmp_dir / "train_split.json"
            validation_split = tmp_dir / "validation_split.json"
            test_split = tmp_dir / "test_split.json"

            samples: list[dict[str, object]] = []
            for idx in range(6):
                image_path = tmp_dir / f"eff_sample_{idx}.jpg"
                self._create_image(image_path)
                samples.append({"path": image_path.as_posix(), "class_index": idx % 3})

            train_split.write_text(json.dumps(samples[:4]), encoding="utf-8")
            validation_split.write_text(json.dumps(samples[4:5]), encoding="utf-8")
            test_split.write_text(json.dumps(samples[3:6]), encoding="utf-8")

            orchestrator = TrainingOrchestrator(
                model_factory=SharedModelFactory(),
                data_loader_factory=SplitJsonLoaderFactory(),
                trainer=BaselineTrainer(),
                evaluator=BaselineEvaluator([f"Class{i}" for i in range(10)]),
                checkpoint_store=JsonCheckpointStore(),
                report_writer=JsonReportWriter(),
            )

            result = orchestrator.run(
                config=config,
                split_artifacts={
                    "train": train_split.as_posix(),
                    "validation": validation_split.as_posix(),
                    "test": test_split.as_posix(),
                },
                output_dir=(tmp_dir / "checkpoints").as_posix(),
                report_output_path=(tmp_dir / "reports" / "efficientnet_stage1_smoke.json").as_posix(),
            )
        finally:
            shutil.rmtree(tmp_dir)

        self.assertIn("accuracy", result)
        self.assertEqual(result["training_state"]["epochs_requested"], config.epochs)


if __name__ == "__main__":
    unittest.main()
