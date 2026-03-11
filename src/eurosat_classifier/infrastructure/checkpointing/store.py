"""Checkpoint persistence for shared training engine."""

import json
from pathlib import Path

import torch


class JsonCheckpointStore:
    """Stores baseline model state and training metadata."""

    def save_best(self, model, training_state: dict[str, object], output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_path / "best_checkpoint.pt"
        metadata_path = output_path / "best_checkpoint.metadata.json"

        torch.save(model.state_dict(), checkpoint_path.as_posix())

        payload = {
            "model_path": checkpoint_path.as_posix(),
            "training_state": training_state,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return checkpoint_path.as_posix()
