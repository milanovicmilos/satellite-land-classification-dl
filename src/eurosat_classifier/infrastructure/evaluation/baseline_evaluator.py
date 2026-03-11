"""Evaluator for baseline model predictions."""

from eurosat_classifier.domain.metrics import MetricSummary
from eurosat_classifier.domain.metrics_calculator import MetricsCalculator


class BaselineEvaluator:
    """Evaluates baseline model by predicting the majority train class."""

    def __init__(self, class_names: list[str]) -> None:
        self._class_names = class_names
        self._calculator = MetricsCalculator()

    def evaluate(self, model, loader):
        y_true = [int(sample["class_index"]) for sample in loader]
        y_pred = [model.majority_class_index for _ in loader]
        summary, _ = self._calculator.calculate(y_true, y_pred, self._class_names)
        return MetricSummary(
            accuracy=summary.accuracy,
            macro_f1_score=summary.macro_f1_score,
            precision=summary.precision,
            recall=summary.recall,
            confusion_matrix=summary.confusion_matrix,
        )
