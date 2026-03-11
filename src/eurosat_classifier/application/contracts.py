"""Application-level contracts used by orchestration code."""

from typing import Protocol

from eurosat_classifier.application.config import TrainingConfig
from eurosat_classifier.domain.entities import DatasetIndex, DatasetSplit, Experiment, PreparedSplit


class ConfigLoader(Protocol):
    """Loads a training configuration from an external source."""

    def load(self, path: str) -> TrainingConfig:
        ...


class TrainingRunner(Protocol):
    """Runs a training workflow for a prepared experiment."""

    def run(self, experiment: Experiment) -> str:
        ...


class DatasetIndexer(Protocol):
    """Builds a validated dataset index from a dataset root path."""

    def build(self, dataset_root: str) -> DatasetIndex:
        ...


class DatasetSplitter(Protocol):
    """Creates deterministic data splits from a dataset index."""

    def split(self, dataset_index: DatasetIndex, split: DatasetSplit) -> PreparedSplit:
        ...


class SplitPersistence(Protocol):
    """Persists split artifacts so all model families can reuse them."""

    def save(self, prepared_split: PreparedSplit, output_dir: str) -> dict[str, str]:
        ...
