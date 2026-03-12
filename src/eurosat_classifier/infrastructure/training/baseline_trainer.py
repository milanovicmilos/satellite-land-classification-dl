"""Baseline trainer implementation with real CNN optimization loop."""

from collections import Counter
from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from eurosat_classifier.domain.metrics_calculator import MetricsCalculator


class BaselineTrainer:
    """Trains baseline CNN with class weights, scheduler, and robust early stopping."""

    def __init__(self, learning_rate: float = 1e-3) -> None:
        self._default_learning_rate = learning_rate
        self._metrics_calculator = MetricsCalculator()

    def train(
        self,
        model,
        loaders,
        epochs: int,
        early_stopping_patience: int,
        learning_rate: float,
        scheduler_factor: float,
        scheduler_patience: int | None,
        min_learning_rate: float,
        early_stopping_min_delta: float,
    ) -> dict[str, object]:
        train_split = loaders["train"]
        if not train_split:
            raise ValueError("Train split is empty; cannot train baseline model.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        class_weights = self._compute_class_weights(train_split, model.num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        effective_learning_rate = learning_rate or self._default_learning_rate
        effective_scheduler_patience = (
            scheduler_patience if scheduler_patience is not None else max(1, early_stopping_patience // 2)
        )

        optimizer = Adam(model.parameters(), lr=effective_learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=max(1, effective_scheduler_patience),
            min_lr=min_learning_rate,
        )

        best_state = deepcopy(model.state_dict())
        best_validation_loss = float("inf")
        best_validation_f1 = float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        epochs_ran = 0
        epoch_logs: list[dict[str, float | int]] = []
        class_names = [f"class_{idx}" for idx in range(model.num_classes)]

        for epoch in range(epochs):
            epochs_ran = epoch + 1
            train_loss = self._train_one_epoch(model, loaders["train"], optimizer, criterion, device)
            validation_loss, validation_accuracy, validation_macro_f1 = self._evaluate_epoch(
                model,
                loaders["validation"],
                criterion,
                device,
                class_names,
            )

            scheduler.step(validation_loss)

            current_lr = float(optimizer.param_groups[0]["lr"])
            epoch_logs.append(
                {
                    "epoch": epochs_ran,
                    "train_loss": float(train_loss),
                    "val_loss": float(validation_loss),
                    "val_acc": float(validation_accuracy),
                    "val_f1": float(validation_macro_f1),
                    "lr": current_lr,
                }
            )

            if validation_macro_f1 > best_validation_f1 + early_stopping_min_delta:
                best_validation_f1 = validation_macro_f1
                best_validation_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                best_epoch = epochs_ran
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                break

        model.load_state_dict(best_state)
        return {
            "epochs_requested": epochs,
            "epochs_ran": epochs_ran,
            "best_validation_loss": float(best_validation_loss),
            "best_validation_f1": float(best_validation_f1),
            "best_epoch": best_epoch,
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "learning_rate": float(effective_learning_rate),
            "scheduler_factor": float(scheduler_factor),
            "scheduler_patience": int(max(1, effective_scheduler_patience)),
            "min_learning_rate": float(min_learning_rate),
            "batch_size": int(getattr(train_split, "batch_size", 0) or 0),
            "class_weights": [float(value) for value in class_weights.detach().cpu().tolist()],
            "epoch_logs": epoch_logs,
        }

    @staticmethod
    def _train_one_epoch(model, train_loader, optimizer, criterion, device) -> float:
        model.train()
        losses: list[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        if not losses:
            raise ValueError("Train loader is empty; cannot compute train loss.")
        return sum(losses) / len(losses)

    def _evaluate_epoch(
        self,
        model,
        validation_loader,
        criterion,
        device,
        class_names: list[str],
    ) -> tuple[float, float, float]:
        model.eval()
        losses: list[float] = []
        y_true: list[int] = []
        y_pred: list[int] = []

        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                losses.append(float(loss.item()))

                predictions = torch.argmax(logits, dim=1)
                y_true.extend(int(value) for value in targets.cpu().tolist())
                y_pred.extend(int(value) for value in predictions.cpu().tolist())

        if not losses:
            raise ValueError("Validation loader is empty; cannot compute validation loss.")

        summary, _ = self._metrics_calculator.calculate(y_true, y_pred, class_names)

        return sum(losses) / len(losses), summary.accuracy, summary.macro_f1_score

    @staticmethod
    def _compute_class_weights(train_loader, class_count: int, device: torch.device) -> torch.Tensor:
        labels = getattr(train_loader.dataset, "labels", None)
        if labels is None:
            raise ValueError("Training dataset does not expose labels for class-weight computation.")

        counts = Counter(int(label) for label in labels)
        total = sum(counts.values())
        if total == 0:
            raise ValueError("Cannot compute class weights from an empty training dataset.")

        weights: list[float] = []
        for class_index in range(class_count):
            class_frequency = counts.get(class_index, 0)
            if class_frequency == 0:
                weights.append(0.0)
            else:
                weights.append(total / (class_count * class_frequency))

        return torch.tensor(weights, dtype=torch.float32, device=device)
