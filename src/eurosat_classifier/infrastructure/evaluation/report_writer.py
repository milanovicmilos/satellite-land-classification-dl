"""Infrastructure report writer for training/evaluation outputs."""

import json
from pathlib import Path
from typing import Any

from eurosat_classifier.domain.metrics import MetricSummary


class JsonReportWriter:
    """Writes evaluation summaries to a JSON report."""

    def write(self, summary: MetricSummary, output_path: str, metadata: dict[str, Any]) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": metadata,
            "metrics": {
                "accuracy": summary.accuracy,
                "macro_f1_score": summary.macro_f1_score,
                "precision": summary.precision,
                "recall": summary.recall,
                "confusion_matrix": summary.confusion_matrix,
            },
        }

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path.as_posix()
