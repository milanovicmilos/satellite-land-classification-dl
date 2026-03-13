"""Tests for shared training orchestrator service."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.application.config import TrainingConfig
from eurosat_classifier.application.services.training_orchestrator import TrainingOrchestrator
from eurosat_classifier.domain.entities import DatasetSplit
from eurosat_classifier.domain.metrics import MetricSummary


class _FakeModelFactory:
    def create(self, model_name: str, model_options: dict[str, object] | None = None):
        return {"model": model_name}


class _FakeDataLoaderFactory:
    def create(
        self,
        split_artifacts: dict[str, str],
        batch_size: int,
        model_name: str | None = None,
        augmentation_mode: str | None = None,
    ) -> dict[str, object]:
        return {"train": [1], "validation": [2], "test": [3]}


class _FakeTrainer:
    def train(
        self,
        model,
        loaders,
        epochs: int,
        early_stopping_patience: int,
        learning_rate: float,
        scheduler_factor: float,
        scheduler_patience: int | None,
        min_learning_rate: float,
        early_stopping_min_delta: float,
    ) -> dict[str, object]:
        return {
            "epochs_ran": 3,
            "best_validation_loss": 0.4,
            "patience": early_stopping_patience,
            "learning_rate": learning_rate,
            "batch_size": 32,
            "epoch_logs": [
                {
                    "epoch": 1,
                    "train_loss": 1.2,
                    "val_loss": 1.1,
                    "val_acc": 0.5,
                    "val_f1": 0.45,
                }
            ],
        }


class _FakeEvaluator:
    def evaluate(self, model, loader):
        return MetricSummary(
            accuracy=0.82,
            macro_f1_score=0.79,
            precision={"A": 0.8},
            recall={"A": 0.78},
            confusion_matrix=[[4, 1], [0, 5]],
        )


class _FakeCheckpointStore:
    def load_checkpoint(self, model, checkpoint_path: str) -> None:
        if isinstance(model, dict):
            model["loaded_from"] = checkpoint_path

    def save_best(self, model, training_state: dict[str, object], output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_file = output_path / "best.ckpt"
        checkpoint_file.write_text("dummy", encoding="utf-8")
        return checkpoint_file.as_posix()


class _FakeReportWriter:
    def write(self, summary, output_path: str, metadata: dict[str, object]) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(metadata), encoding="utf-8")
        return path.as_posix()


class TrainingOrchestratorTests(unittest.TestCase):
    def test_run_returns_training_and_evaluation_outputs(self) -> None:
        orchestrator = TrainingOrchestrator(
            model_factory=_FakeModelFactory(),
            data_loader_factory=_FakeDataLoaderFactory(),
            trainer=_FakeTrainer(),
            evaluator=_FakeEvaluator(),
            checkpoint_store=_FakeCheckpointStore(),
            report_writer=_FakeReportWriter(),
        )

        config = TrainingConfig(
            experiment_name="baseline-run",
            dataset_root="data/EuroSAT",
            model_name="baseline_cnn",
            epochs=10,
            batch_size=32,
            early_stopping_patience=3,
            split=DatasetSplit(0.7, 0.15, 0.15, 42),
            augmentation_mode="none",
        )

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            result = orchestrator.run(
                config=config,
                split_artifacts={
                    "train": "artifacts/splits/train_split.json",
                    "validation": "artifacts/splits/validation_split.json",
                    "test": "artifacts/splits/test_split.json",
                },
                output_dir=(tmp_dir / "checkpoints").as_posix(),
                report_output_path=(tmp_dir / "reports" / "baseline_metrics.json").as_posix(),
            )

            self.assertIn("accuracy", result)
            self.assertIn("checkpoint_path", result)
            self.assertIn("report_path", result)
            self.assertIn("confusion_matrix", result)
            self.assertEqual(result["training_state"]["epochs_ran"], 3)

            report_content = Path(result["report_path"]).read_text(encoding="utf-8")
            self.assertIn("hyperparameters", report_content)
            self.assertIn("epoch_logs", report_content)
        finally:
            shutil.rmtree(tmp_dir)

    def test_resume_checkpoint_is_loaded_before_training(self) -> None:
        class _ResumeAwareTrainer:
            def train(
                self,
                model,
                loaders,
                epochs: int,
                early_stopping_patience: int,
                learning_rate: float,
                scheduler_factor: float,
                scheduler_patience: int | None,
                min_learning_rate: float,
                early_stopping_min_delta: float,
            ):
                loaded_from = model.get("loaded_from", "")
                if not loaded_from.replace("\\", "/").endswith("/stage1/best_checkpoint.pt"):
                    raise AssertionError("Checkpoint must be loaded before training starts.")
                return {
                    "epochs_ran": 1,
                    "best_validation_loss": 0.1,
                    "learning_rate": 0.001,
                    "batch_size": 8,
                    "epoch_logs": [],
                }

        orchestrator = TrainingOrchestrator(
            model_factory=_FakeModelFactory(),
            data_loader_factory=_FakeDataLoaderFactory(),
            trainer=_ResumeAwareTrainer(),
            evaluator=_FakeEvaluator(),
            checkpoint_store=_FakeCheckpointStore(),
            report_writer=_FakeReportWriter(),
        )

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            resume_path = tmp_dir / "stage1" / "best_checkpoint.pt"
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            resume_path.write_text("dummy", encoding="utf-8")

            config = TrainingConfig(
                experiment_name="efficientnet-stage2",
                dataset_root="data/EuroSAT",
                model_name="efficientnet_b0",
                epochs=1,
                batch_size=8,
                early_stopping_patience=1,
                split=DatasetSplit(0.7, 0.15, 0.15, 42),
                augmentation_mode="flips",
                resume_from=resume_path.as_posix(),
                model_options={"freeze_backbone": False},
            )

            result = orchestrator.run(
                config=config,
                split_artifacts={"train": "t.json", "validation": "v.json", "test": "x.json"},
                output_dir=(tmp_dir / "checkpoints").as_posix(),
                report_output_path=(tmp_dir / "reports" / "stage2_metrics.json").as_posix(),
            )
            self.assertIn("accuracy", result)
        finally:
            shutil.rmtree(tmp_dir)

    def test_missing_resume_checkpoint_fails_before_model_initialization(self) -> None:
        class _FailIfModelCreatedFactory:
            def create(self, model_name: str, model_options: dict[str, object] | None = None):
                raise AssertionError("Model must not be created when resume checkpoint is missing.")

        orchestrator = TrainingOrchestrator(
            model_factory=_FailIfModelCreatedFactory(),
            data_loader_factory=_FakeDataLoaderFactory(),
            trainer=_FakeTrainer(),
            evaluator=_FakeEvaluator(),
            checkpoint_store=_FakeCheckpointStore(),
            report_writer=_FakeReportWriter(),
        )

        config = TrainingConfig(
            experiment_name="invalid-resume",
            dataset_root="data/EuroSAT",
            model_name="efficientnet_b0",
            epochs=1,
            batch_size=8,
            early_stopping_patience=1,
            split=DatasetSplit(0.7, 0.15, 0.15, 42),
            augmentation_mode="flips",
            resume_from="checkpoints/does-not-exist.pt",
            model_options={"freeze_backbone": False},
        )

        with self.assertRaises(FileNotFoundError):
            orchestrator.run(
                config=config,
                split_artifacts={"train": "t.json", "validation": "v.json", "test": "x.json"},
                output_dir="tmp/checkpoints",
                report_output_path="tmp/report.json",
            )


if __name__ == "__main__":
    unittest.main()
