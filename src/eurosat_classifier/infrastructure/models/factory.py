"""Shared model factory implementations."""

from eurosat_classifier.infrastructure.models.baseline_cnn import BaselineCnnModel
from eurosat_classifier.infrastructure.models.efficientnet_b0 import EfficientNetB0Model


class SharedModelFactory:
    """Creates model instances supported by the shared training engine."""

    def create(self, model_name: str):
        if model_name == "baseline_cnn":
            return BaselineCnnModel()

        if model_name == "efficientnet_b0":
            return EfficientNetB0Model(use_pretrained=True, freeze_backbone=False)

        if model_name == "resnet50":
            raise NotImplementedError(
                "ResNet50 implementation belongs to Bojan and is out of scope for this phase."
            )

        raise ValueError(f"Unsupported model name: {model_name}")
