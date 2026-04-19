"""Evaluator for baseline model predictions."""

import torch

from eurosat_classifier.domain.metrics import MetricSummary
from eurosat_classifier.domain.metrics_calculator import MetricsCalculator


class BaselineEvaluator:
    """Evaluates baseline CNN on test data and computes required metrics."""

    def __init__(self, class_names: list[str]) -> None:
        self._class_names = class_names
        self._calculator = MetricsCalculator()

    def evaluate(self, model, loader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                logits = model(inputs)
                predictions = torch.argmax(logits, dim=1)

                y_true.extend(int(value) for value in targets.cpu().tolist())
                y_pred.extend(int(value) for value in predictions.cpu().tolist())

        summary, _ = self._calculator.calculate(y_true, y_pred, self._class_names)
        return MetricSummary(
            accuracy=summary.accuracy,
            macro_f1_score=summary.macro_f1_score,
            precision=summary.precision,
            recall=summary.recall,
            confusion_matrix=summary.confusion_matrix,
        )
