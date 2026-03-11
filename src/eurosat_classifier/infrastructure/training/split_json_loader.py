"""Split artifact loader used by training and evaluation flows."""

import json
from pathlib import Path
from typing import Any


class SplitJsonLoaderFactory:
    """Loads split JSON files and returns loader-like collections."""

    def create(self, split_artifacts: dict[str, str], batch_size: int) -> dict[str, Any]:
        del batch_size

        return {
            "train": self._read(split_artifacts["train"]),
            "validation": self._read(split_artifacts["validation"]),
            "test": self._read(split_artifacts["test"]),
        }

    @staticmethod
    def _read(path: str) -> list[dict[str, Any]]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
