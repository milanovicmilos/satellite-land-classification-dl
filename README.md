# EuroSAT Land-Use Classification

Python project for EuroSAT land-use classification with a baseline CNN, EfficientNetB0 fine-tuning, and ResNet50 fine-tuning.

## Project Scope

- Dataset: EuroSAT RGB dataset.
- Task: 10-class land-use classification.
- Evaluation: stratified train/validation/test split with a fixed seed.
- Metrics: accuracy, macro F1-score, confusion matrix, per-class precision, and per-class recall.

## Team Ownership

- Miloš Milanović: EfficientNetB0 and shared infrastructure.
- Bojan Živanić: ResNet50.

## Architecture

The project follows SOLID principles, Clean Architecture, and Clean Code rules.

```text
src/eurosat_classifier/
  domain/          Core entities and business rules
  application/     Use cases, orchestration, and interfaces
  infrastructure/  Filesystem, config loading, model wiring
  entrypoints/     CLI and external execution entrypoints
tests/             Unit tests for pure and orchestration-friendly code
configs/           Example experiment configurations
data/              Local dataset storage, ignored by Git
```

## Verified Local Commands

The following commands are intended to work in the current scaffold.

```powershell
c:/Users/Milos/PythonProjects/satellite-land-classification-dl/.venv/Scripts/python.exe -m unittest discover -s tests
$env:PYTHONPATH='src'; c:/Users/Milos/PythonProjects/satellite-land-classification-dl/.venv/Scripts/python.exe -m eurosat_classifier --help
```

## Notes

- This scaffold intentionally leaves dataset integration and deep learning framework wiring for later tasks.
- The current project specification targets the RGB JPEG dataset in `data/EuroSAT`, not the multispectral TIFF variant.
- ResNet50-specific implementation should remain Bojan's responsibility unless explicitly requested otherwise.
