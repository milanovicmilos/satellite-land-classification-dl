"""Split artifact loader used by training and evaluation flows."""

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset


class SplitJsonDataset(Dataset):
    """Dataset built from persisted split JSON entries."""

    def __init__(self, samples: list[dict[str, Any]], image_size: int = 64) -> None:
        self._samples = samples
        self._image_size = image_size
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

            image_data = torch.tensor(list(image.tobytes()), dtype=torch.uint8)
            image_tensor = image_data.view(self._image_size, self._image_size, 3).permute(2, 0, 1)
            image_tensor = image_tensor.to(dtype=torch.float32)
            image_tensor = image_tensor / 255.0

        label_tensor = torch.tensor(class_index, dtype=torch.long)
        return image_tensor, label_tensor


class SplitJsonLoaderFactory:
    """Loads split JSON files and returns loader-like collections."""

    def create(self, split_artifacts: dict[str, str], batch_size: int) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        train_samples = self._read(split_artifacts["train"])
        validation_samples = self._read(split_artifacts["validation"])
        test_samples = self._read(split_artifacts["test"])

        return {
            "train": DataLoader(
                SplitJsonDataset(train_samples),
                batch_size=batch_size,
                shuffle=True,
            ),
            "validation": DataLoader(
                SplitJsonDataset(validation_samples),
                batch_size=batch_size,
                shuffle=False,
            ),
            "test": DataLoader(
                SplitJsonDataset(test_samples),
                batch_size=batch_size,
                shuffle=False,
            ),
        }

    @staticmethod
    def _read(path: str) -> list[dict[str, Any]]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
