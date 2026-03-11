"""Application configuration objects."""

from dataclasses import dataclass

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

    def to_experiment(self) -> Experiment:
        self.split.validate()
        return Experiment(
            name=self.experiment_name,
            dataset_root=self.dataset_root,
            model_name=self.model_name,
            split=self.split,
        )
