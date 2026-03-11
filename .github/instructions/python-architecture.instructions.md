---
description: "Use when writing or refactoring Python machine learning code, training pipelines, evaluation utilities, dataset loaders, or model modules in this repository. Enforces English-only code comments, SOLID, Clean Architecture, teammate ownership boundaries, and testable design."
name: "Python Architecture Rules"
applyTo: "**/*.py"
---

# Python Architecture Rules

## Language And Readability
- Write all code comments, docstrings, TODOs, and developer-facing text in English.
- Prefer clear names over abbreviations unless the abbreviation is a standard ML term.
- Keep functions and classes focused on a single reason to change.

## Structural Rules
- Separate domain logic from framework-specific code.
- Keep training orchestration outside model classes.
- Keep dataset reading and preprocessing outside training loops.
- Isolate filesystem, logging, checkpointing, and third-party model wiring behind infrastructure components.
- Prefer constructor injection or function parameters over direct imports of concrete dependencies.

## SOLID Expectations
- Single Responsibility: one concern per class or function.
- Open/Closed: add new models or evaluators through extension points, not branching across the codebase.
- Liskov Substitution: keep interchangeable contracts consistent for trainers, datasets, and model factories.
- Interface Segregation: use small protocols or abstract interfaces instead of broad kitchen-sink APIs.
- Dependency Inversion: application workflows should depend on abstractions, not concrete PyTorch or filesystem details.

## ML Project Boundaries
- Shared infrastructure may support baseline CNN, EfficientNetB0, and ResNet50.
- Miloš owns EfficientNetB0-specific implementation.
- Bojan owns ResNet50-specific implementation.
- Unless explicitly requested, do not implement ResNet50-specific code in detail. Create extension points, interfaces, stubs, or configuration placeholders instead.
- If a task touches shared code, keep ownership boundaries explicit in names and comments.

## Testing And Reproducibility
- Prefer pure functions for preprocessing, metric computation, config parsing, and split generation so they are easy to test.
- Validate assumptions at boundaries with clear exceptions.
- Keep randomness controlled through explicit seed plumbing.
- Avoid side effects during import time.

## Preferred Patterns
- Use dataclasses for immutable or structured configuration objects when appropriate.
- Use protocols or abstract base classes for swappable services.
- Keep configuration loading separate from runtime execution.
- Return structured results from training and evaluation instead of printing from deep layers.