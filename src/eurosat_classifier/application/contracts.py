"""Application-level contracts used by orchestration code."""

from typing import Protocol

from eurosat_classifier.application.config import TrainingConfig
from eurosat_classifier.domain.entities import Experiment


class ConfigLoader(Protocol):
    """Loads a training configuration from an external source."""

    def load(self, path: str) -> TrainingConfig:
        ...


class TrainingRunner(Protocol):
    """Runs a training workflow for a prepared experiment."""

    def run(self, experiment: Experiment) -> str:
        ...
