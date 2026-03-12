"""Model registry and discovery helpers for Open/Closed factory behavior."""

from __future__ import annotations

import importlib
from pathlib import Path
import pkgutil
from typing import Any, Callable


ModelBuilder = Callable[[dict[str, object] | None], Any]

SUPPORTED_SHARED_MODELS = {
    "baseline_cnn": "Shared baseline CNN implementation.",
    "efficientnet_b0": "Milos-owned EfficientNetB0 implementation with staged fine-tuning support.",
    "resnet50": "Reserved for Bojan's ResNet50 implementation.",
}

_REGISTERED_BUILDERS: dict[str, ModelBuilder] = {}
_MODELS_DISCOVERED = False


def register_model(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    """Registers a model builder under a model name."""

    def _decorator(builder: ModelBuilder) -> ModelBuilder:
        _REGISTERED_BUILDERS[name] = builder
        return builder

    return _decorator


def discover_model_builders() -> None:
    """Imports model modules to trigger registration decorators."""

    global _MODELS_DISCOVERED
    if _MODELS_DISCOVERED:
        return

    models_dir = Path(__file__).resolve().parent
    ignored = {"__init__", "factory", "registry"}
    for module_info in pkgutil.iter_modules([models_dir.as_posix()]):
        module_name = module_info.name
        if module_name in ignored:
            continue
        importlib.import_module(f"{__package__}.{module_name}")

    _MODELS_DISCOVERED = True


def create_registered_model(model_name: str, model_options: dict[str, object] | None = None) -> Any:
    """Builds a model by looking it up in the discovered registry."""

    discover_model_builders()
    builder = _REGISTERED_BUILDERS.get(model_name)
    if builder is not None:
        return builder(model_options or {})

    if model_name in SUPPORTED_SHARED_MODELS:
        raise NotImplementedError(
            f"Model '{model_name}' is declared but not yet registered with the model registry."
        )

    raise ValueError(f"Unsupported model name: {model_name}")
