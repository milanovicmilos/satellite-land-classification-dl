"""EfficientNetB0 adapter for EuroSAT classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetB0Model(nn.Module):
    """EfficientNetB0 model with optional backbone freezing for staged fine-tuning."""

    def __init__(
        self,
        num_classes: int = 10,
        use_pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        self.backbone = efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

        self.backbone_frozen = False
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freezes or unfreezes feature extractor parameters for staged training."""

        for parameter in self.backbone.features.parameters():
            parameter.requires_grad = trainable
        self.backbone_frozen = not trainable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.backbone(inputs)
