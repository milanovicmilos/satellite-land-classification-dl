"""Shared model factory implementations."""

from eurosat_classifier.infrastructure.models.baseline_cnn import BaselineCnnModel
from eurosat_classifier.infrastructure.models.efficientnet_b0 import EfficientNetB0Model


class SharedModelFactory:
    """Creates model instances supported by the shared training engine."""

    def create(self, model_name: str, model_options: dict[str, object] | None = None):
        options = model_options or {}

        if model_name == "baseline_cnn":
            return BaselineCnnModel()

        if model_name == "efficientnet_b0":
            use_pretrained = bool(options.get("use_pretrained", True))
            freeze_backbone = bool(options.get("freeze_backbone", False))
            return EfficientNetB0Model(
                use_pretrained=use_pretrained,
                freeze_backbone=freeze_backbone,
            )

        if model_name == "resnet50":
            raise NotImplementedError(
                "ResNet50 implementation belongs to Bojan and is out of scope for this phase."
            )

        raise ValueError(f"Unsupported model name: {model_name}")
