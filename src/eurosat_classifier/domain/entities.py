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


@dataclass(frozen=True)
class LabeledSample:
    """Represents one labeled sample from the dataset index."""

    path: str
    class_name: str
    class_index: int


@dataclass(frozen=True)
class DatasetIndex:
    """Represents dataset samples grouped by class."""

    dataset_root: str
    samples_by_class: dict[str, list[LabeledSample]]

    def total_samples(self) -> int:
        return sum(len(samples) for samples in self.samples_by_class.values())

    def total_classes(self) -> int:
        return len(self.samples_by_class)


@dataclass(frozen=True)
class PreparedSplit:
    """Represents a deterministic dataset split for reuse across models."""

    train: list[LabeledSample]
    validation: list[LabeledSample]
    test: list[LabeledSample]
    seed: int
