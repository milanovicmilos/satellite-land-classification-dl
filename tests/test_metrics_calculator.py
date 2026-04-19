"""Tests for domain-level multi-class metrics calculation."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.domain.metrics_calculator import MetricsCalculator


class MetricsCalculatorTests(unittest.TestCase):
    def test_calculate_returns_expected_metrics(self) -> None:
        calculator = MetricsCalculator()

        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 1, 2, 0]
        class_names = ["A", "B", "C"]

        summary, confusion = calculator.calculate(y_true, y_pred, class_names)

        self.assertAlmostEqual(summary.accuracy, 4 / 6)
        self.assertAlmostEqual(summary.macro_f1_score, 0.6555555555, places=6)
        self.assertEqual(confusion, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertIn("A", summary.precision)
        self.assertIn("A", summary.recall)

    def test_calculate_raises_for_invalid_lengths(self) -> None:
        calculator = MetricsCalculator()

        with self.assertRaises(ValueError):
            calculator.calculate([0], [0, 1], ["A", "B"])

    def test_calculate_raises_for_empty_class_names(self) -> None:
        calculator = MetricsCalculator()

        with self.assertRaises(ValueError):
            calculator.calculate([0, 1], [0, 1], [])

    def test_calculate_raises_for_invalid_prediction_index(self) -> None:
        calculator = MetricsCalculator()

        with self.assertRaises(ValueError):
            calculator.calculate([0, 1], [0, 3], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
