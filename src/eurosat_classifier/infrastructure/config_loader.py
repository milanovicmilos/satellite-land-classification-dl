"""Infrastructure implementation for JSON config loading."""

import json
from pathlib import Path
from typing import Any

from eurosat_classifier.application.config import TrainingConfig
from eurosat_classifier.domain.entities import DatasetSplit


class JsonConfigLoader:
    """Loads training configuration from a JSON file."""

    def __init__(self, defaults_path: str | None = None) -> None:
        self._defaults_path = defaults_path

    @staticmethod
    def _read_json(path: str) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = JsonConfigLoader._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _resolve_augmentation_mode(raw_config: dict[str, Any]) -> str | None:
        training_config = raw_config.get("training", {})
        resolved_mode = training_config.get("augmentation_mode")
        model_name = str(raw_config.get("model", {}).get("name", "")).lower()

        if resolved_mode is None and model_name.startswith("efficientnet"):
            return "full"

        return resolved_mode

    def load(self, path: str) -> TrainingConfig:
        config_data: dict[str, Any] = {}

        if self._defaults_path:
            config_data = self._read_json(self._defaults_path)

        user_config = self._read_json(path)
        raw_config = self._deep_merge(config_data, user_config)

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
            learning_rate=raw_config["training"].get("learning_rate", 1e-3),
            scheduler_factor=raw_config["training"].get("scheduler_factor", 0.5),
            scheduler_patience=raw_config["training"].get("scheduler_patience"),
            min_learning_rate=raw_config["training"].get("min_learning_rate", 1e-6),
            early_stopping_min_delta=raw_config["training"].get("early_stopping_min_delta", 0.0),
            augmentation_mode=self._resolve_augmentation_mode(raw_config),
            split=split,
            resume_from=raw_config["training"].get("resume_from"),
            model_options=raw_config["model"].get("options", {}),
        )
