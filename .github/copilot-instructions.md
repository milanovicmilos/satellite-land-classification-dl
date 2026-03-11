# Project Guidelines

## Project Context
- This repository is a Python project for EuroSAT land-use classification.
- Current source of truth is `specifikacija.txt` until code, README, and configs are added.
- Planned models are baseline CNN, EfficientNetB0 fine-tuning, and ResNet50 fine-tuning.

## Collaboration Rules
- This is a two-member project.
- Miloš Milanović owns EfficientNetB0 work and may also work on shared infrastructure, evaluation, documentation, dataset handling, and baseline CNN support.
- Bojan Živanić owns ResNet50 implementation.
- Do not implement both advanced models in the same task by default.
- Unless the user explicitly asks otherwise, avoid generating ResNet50 training/model implementation for Miloš; prefer abstractions, interfaces, configuration hooks, or placeholders that preserve Bojan's ownership boundary.

## Code Style
- All code, comments, docstrings, configuration comments, commit-ready prompts, and generated documentation must be in English.
- Prefer explicit, readable names and small focused units.
- Use type hints on public functions and non-trivial internal functions.
- Prefer dataclasses, protocols, and pure functions where they reduce coupling and improve testability.
- Avoid hidden global state, hard-coded paths, and tightly coupled training scripts.

## Architecture
- Favor SOLID principles, Clean Architecture, and Clean Code practices.
- Keep domain logic independent from frameworks.
- Separate responsibilities into clear layers when code is added:
  - domain: entities, value objects, business rules, metrics contracts
  - application: training/evaluation use cases, orchestration, interfaces
  - infrastructure: dataset loading, model backbones, filesystem, logging, checkpointing
  - entrypoints: CLI scripts, config loading, experiment runners
- Keep model definition, data access, training loop, and evaluation/reporting separate.
- Prefer dependency inversion for components such as datasets, trainers, evaluators, checkpoint stores, and model factories.

## Build And Test
- No verified build, training, or test commands exist yet.
- Do not present guessed commands as established project workflow.
- If scaffolding is added, keep commands reproducible, minimal, and documented.
- Prefer deterministic experiments: fixed seed, explicit dataset split strategy, and saved configuration.

## Working Conventions
- Before making broad structural changes, inspect the current repository and preserve user-authored work.
- Ask before changing project scope, dataset assumptions, metric definitions, experiment tracking approach, or model ownership boundaries.
- When adding new files, prefer a structure that can scale to baseline CNN plus one owned fine-tuned model without forcing the second teammate's implementation.
- When writing implementation plans, distinguish shared infrastructure from model-specific work.

## Key Files
- `specifikacija.txt`: project requirements, dataset split, target metrics, and assigned models.
- `.github/instructions/python-architecture.instructions.md`: Python implementation rules.
- `.github/instructions/project-docs.instructions.md`: documentation and planning rules.
