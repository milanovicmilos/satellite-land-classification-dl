"""Baseline CNN model for EuroSAT RGB classification."""

import torch
from torch import nn


class BaselineCnnModel(nn.Module):
    """Simple CNN trained from scratch as the project baseline."""

    def __init__(self, input_channels: int = 3, input_size: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.input_size = input_size
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        reduced_size = input_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced_size * reduced_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(features)
