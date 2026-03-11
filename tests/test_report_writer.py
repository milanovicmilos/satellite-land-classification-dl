"""Tests for shared JSON report writer."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.domain.metrics import MetricSummary
from eurosat_classifier.infrastructure.evaluation.report_writer import JsonReportWriter


class JsonReportWriterTests(unittest.TestCase):
    def test_write_creates_json_report(self) -> None:
        writer = JsonReportWriter()
        summary = MetricSummary(
            accuracy=0.9,
            macro_f1_score=0.8,
            precision={"A": 0.9},
            recall={"A": 0.8},
        )

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            output_path = tmp_dir / "reports" / "baseline.json"
            result = writer.write(summary, output_path.as_posix(), {"model_name": "baseline_cnn"})

            self.assertTrue(Path(result).exists())
            content = Path(result).read_text(encoding="utf-8")
            self.assertIn('"accuracy": 0.9', content)
            self.assertIn('"model_name": "baseline_cnn"', content)
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
