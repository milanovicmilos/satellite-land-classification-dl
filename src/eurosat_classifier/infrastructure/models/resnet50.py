"""ResNet50 adapter for EuroSAT classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from eurosat_classifier.infrastructure.models.registry import register_model


class ResNet50Model(nn.Module):
    """ResNet50 model with optional backbone freezing for staged fine-tuning."""

    def __init__(
        self,
        num_classes: int = 10,
        use_pretrained: bool = False,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        weights = ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        self.backbone = resnet50(weights=weights)

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        self.backbone_frozen = False
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def train(self, mode: bool = True):
        """Keeps backbone BatchNorm layers in eval mode while the backbone is frozen."""

        super().train(mode)
        if self.backbone_frozen:
            self._set_backbone_batchnorm_eval()
        return self

    def _set_backbone_batchnorm_eval(self) -> None:
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freezes or unfreezes feature extractor parameters for staged training."""

        for name, parameter in self.backbone.named_parameters():
            if name.startswith("fc."):
                continue
            parameter.requires_grad = trainable

        for parameter in self.backbone.fc.parameters():
            parameter.requires_grad = True

        self.backbone_frozen = not trainable
        if not trainable:
            self._set_backbone_batchnorm_eval()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)


@register_model("resnet50")
def build_resnet50_model(
    model_options: dict[str, object] | None = None,
) -> ResNet50Model:
    """Builds ResNet50 with config-driven options for staged fine-tuning."""

    options = model_options or {}
    use_pretrained = bool(options.get("use_pretrained", False))
    freeze_backbone = bool(options.get("freeze_backbone", False))
    return ResNet50Model(
        use_pretrained=use_pretrained,
        freeze_backbone=freeze_backbone,
    )
