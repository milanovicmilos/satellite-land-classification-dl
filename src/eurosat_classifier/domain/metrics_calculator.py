"""Domain-level metric calculations for multi-class classification."""

from eurosat_classifier.domain.metrics import MetricSummary


class MetricsCalculator:
    """Computes required evaluation metrics from class predictions."""

    def calculate(
        self,
        y_true: list[int],
        y_pred: list[int],
        class_names: list[str],
    ) -> tuple[MetricSummary, list[list[int]]]:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        if not class_names:
            raise ValueError("class_names cannot be empty.")

        class_count = len(class_names)
        confusion_matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]

        for true_idx, pred_idx in zip(y_true, y_pred):
            if true_idx < 0 or true_idx >= class_count:
                raise ValueError(f"y_true contains invalid class index: {true_idx}")
            if pred_idx < 0 or pred_idx >= class_count:
                raise ValueError(f"y_pred contains invalid class index: {pred_idx}")
            confusion_matrix[true_idx][pred_idx] += 1

        total = len(y_true)
        correct = sum(confusion_matrix[i][i] for i in range(class_count))
        accuracy = (correct / total) if total else 0.0

        precision: dict[str, float] = {}
        recall: dict[str, float] = {}
        f1_scores: list[float] = []

        for idx, class_name in enumerate(class_names):
            true_positive = confusion_matrix[idx][idx]
            false_positive = sum(confusion_matrix[row][idx] for row in range(class_count)) - true_positive
            false_negative = sum(confusion_matrix[idx][col] for col in range(class_count)) - true_positive

            precision_denominator = true_positive + false_positive
            recall_denominator = true_positive + false_negative

            class_precision = (
                true_positive / precision_denominator if precision_denominator else 0.0
            )
            class_recall = true_positive / recall_denominator if recall_denominator else 0.0

            precision[class_name] = class_precision
            recall[class_name] = class_recall

            if (class_precision + class_recall) == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(
                    2.0 * class_precision * class_recall / (class_precision + class_recall)
                )

        macro_f1_score = sum(f1_scores) / class_count

        summary = MetricSummary(
            accuracy=accuracy,
            macro_f1_score=macro_f1_score,
            precision=precision,
            recall=recall,
        )
        return summary, confusion_matrix
