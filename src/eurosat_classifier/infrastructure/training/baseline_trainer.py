"""Baseline trainer implementation with real CNN optimization loop."""

from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam


class BaselineTrainer:
    """Trains baseline CNN with early stopping on validation loss."""

    def train(
        self,
        model,
        loaders,
        epochs: int,
        early_stopping_patience: int,
    ) -> dict[str, object]:
        train_split = loaders["train"]
        if not train_split:
            raise ValueError("Train split is empty; cannot train baseline model.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=1e-3)

        best_state = deepcopy(model.state_dict())
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        epochs_ran = 0

        for epoch in range(epochs):
            epochs_ran = epoch + 1
            self._train_one_epoch(model, loaders["train"], optimizer, criterion, device)
            validation_loss = self._evaluate_loss(model, loaders["validation"], criterion, device)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = deepcopy(model.state_dict())
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
            "early_stopping_patience": early_stopping_patience,
        }

    @staticmethod
    def _train_one_epoch(model, train_loader, optimizer, criterion, device) -> None:
        model.train()
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

    @staticmethod
    def _evaluate_loss(model, validation_loader, criterion, device) -> float:
        model.eval()
        losses: list[float] = []

        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                loss = criterion(logits, targets)
                losses.append(float(loss.item()))

        if not losses:
            raise ValueError("Validation loader is empty; cannot compute validation loss.")

        return sum(losses) / len(losses)
