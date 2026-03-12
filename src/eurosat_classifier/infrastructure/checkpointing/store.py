"""Checkpoint persistence for shared training engine."""

import json
from pathlib import Path

import torch


class JsonCheckpointStore:
    """Stores baseline model state and training metadata."""

    @staticmethod
    def load_checkpoint(model, checkpoint_path: str) -> None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)

    def save_best(self, model, training_state: dict[str, object], output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_path / "best_checkpoint.pt"
        metadata_path = output_path / "best_checkpoint.metadata.json"

        torch.save(model.state_dict(), checkpoint_path.as_posix())

        payload = {
            "model_path": checkpoint_path.as_posix(),
            "hyperparameters": {
                "learning_rate": training_state.get("learning_rate"),
                "batch_size": training_state.get("batch_size"),
                "epochs_requested": training_state.get("epochs_requested"),
                "epochs_ran": training_state.get("epochs_ran"),
                "early_stopping_patience": training_state.get("early_stopping_patience"),
            },
            "training_state": training_state,
        }
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return checkpoint_path.as_posix()
