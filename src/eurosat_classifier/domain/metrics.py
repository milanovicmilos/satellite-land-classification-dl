"""Domain contracts for evaluation metrics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSummary:
    """Carries evaluation outputs in a framework-agnostic way."""

    accuracy: float
    macro_f1_score: float
    precision: dict[str, float]
    recall: dict[str, float]
    confusion_matrix: list[list[int]] | None = None
