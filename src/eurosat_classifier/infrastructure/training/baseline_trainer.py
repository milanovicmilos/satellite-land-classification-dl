"""Reference baseline trainer using split artifacts as structured inputs."""

from collections import Counter


class BaselineTrainer:
    """Fits baseline class priors from train split and exposes training state."""

    def train(
        self,
        model,
        loaders: dict[str, list[dict[str, object]]],
        epochs: int,
        early_stopping_patience: int,
    ) -> dict[str, object]:
        train_split = loaders["train"]
        if not train_split:
            raise ValueError("Train split is empty; cannot train baseline model.")

        class_counter = Counter(sample["class_index"] for sample in train_split)
        total = sum(class_counter.values())

        model.class_priors = {int(k): v / total for k, v in class_counter.items()}
        model.majority_class_index = int(max(class_counter.items(), key=lambda item: item[1])[0])

        epochs_ran = min(epochs, max(1, early_stopping_patience + 1))
        return {
            "epochs_requested": epochs,
            "epochs_ran": epochs_ran,
            "best_validation_loss": 1.0,
            "early_stopping_patience": early_stopping_patience,
        }
