"""Shared training orchestration service for baseline and fine-tuned models."""

from pathlib import Path
from typing import Any

from eurosat_classifier.application.config import TrainingConfig
from eurosat_classifier.application.contracts import (
    CheckpointStore,
    DataLoaderFactory,
    Evaluator,
    ModelFactory,
    ReportWriter,
    Trainer,
)
from eurosat_classifier.domain.metrics import MetricSummary
from eurosat_classifier.infrastructure.reproducibility import set_seed


class TrainingOrchestrator:
    """Coordinates model creation, training, evaluation, checkpointing, and reporting."""

    def __init__(
        self,
        model_factory: ModelFactory,
        data_loader_factory: DataLoaderFactory,
        trainer: Trainer,
        evaluator: Evaluator,
        checkpoint_store: CheckpointStore,
        report_writer: ReportWriter,
    ) -> None:
        self._model_factory = model_factory
        self._data_loader_factory = data_loader_factory
        self._trainer = trainer
        self._evaluator = evaluator
        self._checkpoint_store = checkpoint_store
        self._report_writer = report_writer

    def run(
        self,
        config: TrainingConfig,
        split_artifacts: dict[str, str],
        output_dir: str,
        report_output_path: str,
    ) -> dict[str, Any]:
        set_seed(config.split.seed)

        if config.resume_from:
            resume_path = Path(config.resume_from)
            if not resume_path.exists():
                raise FileNotFoundError(
                    "Invalid training configuration: resume_from checkpoint path does not exist: "
                    f"{config.resume_from}"
                )

        model = self._model_factory.create(config.model_name, config.model_options)
        if config.resume_from:
            self._checkpoint_store.load_checkpoint(model, config.resume_from)

        loaders = self._data_loader_factory.create(
            split_artifacts,
            config.batch_size,
            model_name=config.model_name,
            augmentation_mode=config.augmentation_mode,
        )

        training_state = self._trainer.train(
            model=model,
            loaders=loaders,
            epochs=config.epochs,
            early_stopping_patience=config.early_stopping_patience,
            learning_rate=config.learning_rate,
            scheduler_factor=config.scheduler_factor,
            scheduler_patience=config.scheduler_patience,
            min_learning_rate=config.min_learning_rate,
            early_stopping_min_delta=config.early_stopping_min_delta,
        )

        test_loader = loaders["test"]
        summary = self._evaluator.evaluate(model, test_loader)

        checkpoint_path = self._checkpoint_store.save_best(
            model=model,
            training_state=training_state,
            output_dir=output_dir,
        )

        report_path = self._report_writer.write(
            summary=summary,
            output_path=report_output_path,
            metadata={
                "model_name": config.model_name,
                "dataset_root": config.dataset_root,
                "split_seed": config.split.seed,
                "checkpoint_path": checkpoint_path,
                "hyperparameters": {
                    "epochs": config.epochs,
                    "batch_size": config.batch_size,
                    "early_stopping_patience": config.early_stopping_patience,
                    "learning_rate": config.learning_rate,
                    "scheduler_factor": config.scheduler_factor,
                    "scheduler_patience": config.scheduler_patience,
                    "min_learning_rate": config.min_learning_rate,
                    "early_stopping_min_delta": config.early_stopping_min_delta,
                    "augmentation_mode": config.augmentation_mode,
                },
                "training_state": training_state,
            },
        )

        return self._build_result(summary, checkpoint_path, report_path, training_state)

    @staticmethod
    def _build_result(
        summary: MetricSummary,
        checkpoint_path: str,
        report_path: str,
        training_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "accuracy": summary.accuracy,
            "macro_f1_score": summary.macro_f1_score,
            "precision": summary.precision,
            "recall": summary.recall,
            "confusion_matrix": summary.confusion_matrix,
            "checkpoint_path": checkpoint_path,
            "report_path": report_path,
            "training_state": training_state,
        }
