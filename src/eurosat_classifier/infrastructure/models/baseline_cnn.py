"""Baseline model representation used by the shared training engine."""

from dataclasses import dataclass, field


@dataclass
class BaselineCnnModel:
    """Holds baseline model state used by the initial reference pipeline.

    Note:
    This phase keeps the model framework-agnostic so shared training engine
    contracts can be validated independently of deep-learning dependencies.
    """

    input_channels: int = 3
    input_size: int = 64
    num_classes: int = 10
    class_priors: dict[int, float] = field(default_factory=dict)
    majority_class_index: int = 0
