"""Split artifact loader used by training and evaluation flows."""

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from eurosat_classifier.infrastructure.models.registry import get_model_normalization


class SplitJsonDataset(Dataset):
    """Dataset built from persisted split JSON entries."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        image_size: int = 64,
        normalize_mean: tuple[float, float, float] | None = None,
        normalize_std: tuple[float, float, float] | None = None,
        augmentation_mode: str = "none",
    ) -> None:
        self._samples = samples
        self._image_size = image_size
        self._normalize_mean = normalize_mean
        self._normalize_std = normalize_std
        self._augmentation_mode = augmentation_mode
        self._transform = self._build_transform()
        self.labels: list[int] = [int(sample["class_index"]) for sample in samples]

    def _build_transform(self) -> T.Compose:
        transforms: list[Any] = [
            T.Resize(
                (self._image_size, self._image_size),
                interpolation=InterpolationMode.BILINEAR,
            )
        ]

        if self._augmentation_mode in {"flips", "full"}:
            transforms.extend(
                [
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomVerticalFlip(p=0.5),
                ]
            )

        if self._augmentation_mode == "full":
            transforms.extend(
                [
                    T.RandomRotation(
                        degrees=20,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=(255, 255, 255),
                    ),
                    T.RandomAffine(
                        degrees=0,
                        translate=(0.08, 0.08),
                        scale=(0.9, 1.1),
                        interpolation=InterpolationMode.BILINEAR,
                        fill=(255, 255, 255),
                    ),
                ]
            )

        transforms.append(T.ToTensor())
        if self._normalize_mean is not None and self._normalize_std is not None:
            transforms.append(T.Normalize(mean=self._normalize_mean, std=self._normalize_std))

        return T.Compose(transforms)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._samples[index]
        image_path = str(sample["path"])
        class_index = int(sample["class_index"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self._transform(image)

        label_tensor = torch.tensor(class_index, dtype=torch.long)
        return image_tensor, label_tensor


class SplitJsonLoaderFactory:
    """Loads split JSON files and returns loader-like collections."""

    def create(
        self,
        split_artifacts: dict[str, str],
        batch_size: int,
        model_name: str | None = None,
        augmentation_mode: str | None = None,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        train_samples = self._read(split_artifacts["train"])
        validation_samples = self._read(split_artifacts["validation"])
        test_samples = self._read(split_artifacts["test"])

        normalize_stats = get_model_normalization(model_name)
        normalize_mean = normalize_stats[0] if normalize_stats else None
        normalize_std = normalize_stats[1] if normalize_stats else None
        training_augmentation_mode = self._resolve_augmentation_mode(model_name, augmentation_mode)

        return {
            "train": DataLoader(
                SplitJsonDataset(
                    train_samples,
                    normalize_mean=normalize_mean,
                    normalize_std=normalize_std,
                    augmentation_mode=training_augmentation_mode,
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

    @staticmethod
    def _resolve_augmentation_mode(model_name: str | None, augmentation_mode: str | None) -> str:
        if augmentation_mode is not None:
            if augmentation_mode not in {"none", "flips", "full"}:
                raise ValueError("augmentation_mode must be one of: none, flips, full")
            return augmentation_mode

        return "full" if model_name in {"efficientnet_b0", "resnet50"} else "none"
