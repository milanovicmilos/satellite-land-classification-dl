"""Infrastructure implementation for JSON config loading."""

import json
from pathlib import Path

from eurosat_classifier.application.config import TrainingConfig
from eurosat_classifier.domain.entities import DatasetSplit


class JsonConfigLoader:
    """Loads training configuration from a JSON file."""

    def load(self, path: str) -> TrainingConfig:
        raw_config = json.loads(Path(path).read_text(encoding="utf-8"))

        split = DatasetSplit(
            train_ratio=raw_config["split"]["train_ratio"],
            validation_ratio=raw_config["split"]["validation_ratio"],
            test_ratio=raw_config["split"]["test_ratio"],
            seed=raw_config["split"]["seed"],
            stratified=raw_config["split"].get("stratified", True),
        )

        return TrainingConfig(
            experiment_name=raw_config["experiment_name"],
            dataset_root=raw_config["dataset_root"],
            model_name=raw_config["model"]["name"],
            epochs=raw_config["training"]["epochs"],
            batch_size=raw_config["training"]["batch_size"],
            early_stopping_patience=raw_config["training"]["early_stopping_patience"],
            split=split,
        )
