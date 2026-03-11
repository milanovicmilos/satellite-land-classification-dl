"""Persistence for deterministic split artifacts."""

import json
from pathlib import Path

from eurosat_classifier.domain.entities import LabeledSample, PreparedSplit


class JsonSplitPersistence:
    """Writes split outputs as JSON files for reuse by model training pipelines."""

    def save(self, prepared_split: PreparedSplit, output_dir: str) -> dict[str, str]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_path = output_path / "train_split.json"
        validation_path = output_path / "validation_split.json"
        test_path = output_path / "test_split.json"
        manifest_path = output_path / "split_manifest.json"

        self._write_samples(train_path, prepared_split.train)
        self._write_samples(validation_path, prepared_split.validation)
        self._write_samples(test_path, prepared_split.test)

        summary_payload = {
            "seed": prepared_split.seed,
            "train_samples": len(prepared_split.train),
            "validation_samples": len(prepared_split.validation),
            "test_samples": len(prepared_split.test),
        }
        manifest_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        return {
            "train": train_path.as_posix(),
            "validation": validation_path.as_posix(),
            "test": test_path.as_posix(),
            "manifest": manifest_path.as_posix(),
        }

    @staticmethod
    def _write_samples(path: Path, samples: list[LabeledSample]) -> None:
        payload = [
            {
                "path": sample.path,
                "class_name": sample.class_name,
                "class_index": sample.class_index,
            }
            for sample in samples
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
