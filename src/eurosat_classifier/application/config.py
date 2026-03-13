"""Application configuration objects."""

from dataclasses import dataclass, field

from eurosat_classifier.domain.entities import DatasetSplit, Experiment


@dataclass(frozen=True)
class TrainingConfig:
    """Defines training parameters shared by entrypoints and use cases."""

    experiment_name: str
    dataset_root: str
    model_name: str
    epochs: int
    batch_size: int
    early_stopping_patience: int
    split: DatasetSplit
    learning_rate: float = 1e-3
    scheduler_factor: float = 0.5
    scheduler_patience: int | None = None
    min_learning_rate: float = 1e-6
    early_stopping_min_delta: float = 0.0
    augmentation_mode: str | None = None
    resume_from: str | None = None
    model_options: dict[str, object] = field(default_factory=dict)

    def to_experiment(self) -> Experiment:
        self.split.validate()
        return Experiment(
            name=self.experiment_name,
            dataset_root=self.dataset_root,
            model_name=self.model_name,
            split=self.split,
        )
