"""Shared model factory implementations."""

from eurosat_classifier.infrastructure.models.registry import create_registered_model


class SharedModelFactory:
    """Creates model instances supported by the shared training engine."""

    def create(self, model_name: str, model_options: dict[str, object] | None = None):
        return create_registered_model(model_name, model_options)
