"""Split artifact loader used by training and evaluation flows."""

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from eurosat_classifier.infrastructure.models.registry import get_model_normalization


class SplitJsonDataset(Dataset):
    """Dataset built from persisted split JSON entries."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        image_size: int = 64,
        normalize_mean: tuple[float, float, float] | None = None,
        normalize_std: tuple[float, float, float] | None = None,
    ) -> None:
        self._samples = samples
        self._image_size = image_size
        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std
        self.labels: list[int] = [int(sample["class_index"]) for sample in samples]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[index]
        image_path = str(sample["path"])
        class_index = int(sample["class_index"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if image.size != (self._image_size, self._image_size):
                image = image.resize((self._image_size, self._image_size))

            image_array = np.array(image, copy=True)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).to(dtype=torch.float32) / 255.0

            if self._normalize_mean is not None and self._normalize_std is not None:
                mean = torch.tensor(self._normalize_mean, dtype=torch.float32).view(3, 1, 1)
                std = torch.tensor(self._normalize_std, dtype=torch.float32).view(3, 1, 1)
                image_tensor = (image_tensor - mean) / std

        label_tensor = torch.tensor(class_index, dtype=torch.long)
        return image_tensor, label_tensor


class SplitJsonLoaderFactory:
    """Loads split JSON files and returns loader-like collections."""

    def create(
        self,
        split_artifacts: dict[str, str],
        batch_size: int,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        train_samples = self._read(split_artifacts["train"])
        validation_samples = self._read(split_artifacts["validation"])
        test_samples = self._read(split_artifacts["test"])

        normalize_stats = get_model_normalization(model_name)
        normalize_mean = normalize_stats[0] if normalize_stats else None
        normalize_std = normalize_stats[1] if normalize_stats else None

        return {
            "train": DataLoader(
                SplitJsonDataset(
                    train_samples,
                    normalize_mean=normalize_mean,
                    normalize_std=normalize_std,
                ),
                batch_size=batch_size,
                shuffle=True,
            ),
            "validation": DataLoader(
                SplitJsonDataset(
                    validation_samples,
                    normalize_mean=normalize_mean,
                    normalize_std=normalize_std,
                ),
                batch_size=batch_size,
                shuffle=False,
            ),
            "test": DataLoader(
                SplitJsonDataset(
                    test_samples,
                    normalize_mean=normalize_mean,
                    normalize_std=normalize_std,
                ),
                batch_size=batch_size,
                shuffle=False,
            ),
        }

    @staticmethod
    def _read(path: str) -> list[dict[str, Any]]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
