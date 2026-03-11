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
    def create(self, model_name: str):
        return {"model": model_name}


class _FakeDataLoaderFactory:
    def create(self, split_artifacts: dict[str, str], batch_size: int) -> dict[str, object]:
        return {"train": [1], "validation": [2], "test": [3]}


class _FakeTrainer:
    def train(self, model, loaders, epochs: int, early_stopping_patience: int) -> dict[str, object]:
        return {
            "epochs_ran": 3,
            "best_validation_loss": 0.4,
            "patience": early_stopping_patience,
        }


class _FakeEvaluator:
    def evaluate(self, model, loader):
        return MetricSummary(
            accuracy=0.82,
            macro_f1_score=0.79,
            precision={"A": 0.8},
            recall={"A": 0.78},
        )


class _FakeCheckpointStore:
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
        path.write_text("report", encoding="utf-8")
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
            self.assertEqual(result["training_state"]["epochs_ran"], 3)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
