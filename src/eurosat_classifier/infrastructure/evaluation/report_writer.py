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

        training_state = metadata.get("training_state", {})
        epoch_logs = training_state.get("epoch_logs", []) if isinstance(training_state, dict) else []

        epoch_log_path = path.with_suffix(".training_log.tmp.json")
        epoch_log_payload = {
            "model_name": metadata.get("model_name"),
            "dataset_root": metadata.get("dataset_root"),
            "split_seed": metadata.get("split_seed"),
            "epoch_logs": epoch_logs,
        }
        epoch_log_path.write_text(json.dumps(epoch_log_payload, indent=2), encoding="utf-8")

        payload = {
            "metadata": metadata,
            "epoch_log_path": epoch_log_path.as_posix(),
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
