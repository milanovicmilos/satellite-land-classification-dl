"""Domain entities for experiments and dataset splitting."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSplit:
    """Represents a reproducible dataset split configuration."""

    train_ratio: float
    validation_ratio: float
    test_ratio: float
    seed: int
    stratified: bool = True

    def validate(self) -> None:
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if round(total, 5) != 1.0:
            raise ValueError("Dataset split ratios must sum to 1.0.")


@dataclass(frozen=True)
class Experiment:
    """Represents a high-level experiment definition."""

    name: str
    dataset_root: str
    model_name: str
    split: DatasetSplit
