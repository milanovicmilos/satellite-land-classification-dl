"""Shared model factory implementations."""

from eurosat_classifier.infrastructure.models.baseline_cnn import BaselineCnnModel


class SharedModelFactory:
    """Creates model instances supported by the shared training engine."""

    def create(self, model_name: str):
        if model_name == "baseline_cnn":
            return BaselineCnnModel()

        if model_name == "efficientnet_b0":
            raise NotImplementedError(
                "EfficientNetB0 implementation belongs to Milos and is scheduled for phase 5."
            )

        if model_name == "resnet50":
            raise NotImplementedError(
                "ResNet50 implementation belongs to Bojan and is out of scope for this phase."
            )

        raise ValueError(f"Unsupported model name: {model_name}")
