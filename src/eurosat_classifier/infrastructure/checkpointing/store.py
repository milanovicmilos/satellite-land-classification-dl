"""Checkpoint persistence for shared training engine."""

import json
from pathlib import Path


class JsonCheckpointStore:
    """Stores baseline model state and training state as JSON checkpoint."""

    def save_best(self, model, training_state: dict[str, object], output_dir: str) -> str:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_path / "best_checkpoint.json"
        payload = {
            "model_state": {
                "majority_class_index": model.majority_class_index,
                "class_priors": model.class_priors,
            },
            "training_state": training_state,
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return checkpoint_path.as_posix()
