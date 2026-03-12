"""Application-level contracts used by orchestration code."""

from typing import Any, Protocol

from eurosat_classifier.application.config import TrainingConfig
from eurosat_classifier.domain.entities import DatasetIndex, DatasetSplit, Experiment, PreparedSplit
from eurosat_classifier.domain.metrics import MetricSummary


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


class ModelFactory(Protocol):
    """Builds model instances by model name."""

    def create(self, model_name: str, model_options: dict[str, Any] | None = None) -> Any:
        ...


class DataLoaderFactory(Protocol):
    """Creates train/validation/test loaders from split artifact files."""

    def create(
        self,
        split_artifacts: dict[str, str],
        batch_size: int,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        ...


class Trainer(Protocol):
    """Runs training with early stopping and returns training artifacts."""

    def train(
        self,
        model: Any,
        loaders: dict[str, Any],
        epochs: int,
        early_stopping_patience: int,
    ) -> dict[str, Any]:
        ...


class Evaluator(Protocol):
    """Evaluates a model and returns metric summaries."""

    def evaluate(self, model: Any, loader: Any) -> MetricSummary:
        ...


class CheckpointStore(Protocol):
    """Stores the best model checkpoint and returns output path."""

    def save_best(self, model: Any, training_state: dict[str, Any], output_dir: str) -> str:
        ...


class ReportWriter(Protocol):
    """Writes model evaluation outputs to report files."""

    def write(self, summary: MetricSummary, output_path: str, metadata: dict[str, Any]) -> str:
        ...
