# EuroSAT Land-Use Classification

Python project for EuroSAT land-use classification with a baseline CNN, EfficientNetB0 fine-tuning, and ResNet50 fine-tuning.

## Project Scope

- Dataset: EuroSAT RGB dataset.
- Task: 10-class land-use classification.
- Evaluation: stratified train/validation/test split with a fixed seed.
- Metrics: accuracy, macro F1-score, confusion matrix, per-class precision, and per-class recall.

## Methodology

- Baseline full run uses `early_stopping_patience=10` as anti-overfitting protection.
- This allows convergence while preserving scientific validity by stopping when Validation Macro F1-score stops improving.

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
$env:PYTHONPATH='src'; c:/Users/Milos/PythonProjects/satellite-land-classification-dl/.venv/Scripts/python.exe -m eurosat_classifier --dry-run --config configs/baseline_cnn.json
$env:PYTHONPATH='src'; c:/Users/Milos/PythonProjects/satellite-land-classification-dl/.venv/Scripts/python.exe -m eurosat_classifier --prepare-dataset --config configs/baseline_cnn.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits
$env:PYTHONPATH='src'; c:/Users/Milos/PythonProjects/satellite-land-classification-dl/.venv/Scripts/python.exe -m eurosat_classifier --run-baseline --config configs/baseline_cnn.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits --reports-output artifacts/reports/baseline_cnn.json --checkpoints-output checkpoints/baseline_cnn

$env:PYTHONPATH='src'; c:/Users/Milos/PythonProjects/satellite-land-classification-dl/.venv/Scripts/python.exe run.py --run-baseline --config configs/efficientnet_b0.stage1.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits --reports-output artifacts/reports/efficientnet_b0_stage1_final.json --checkpoints-output checkpoints/efficientnet_b0/stage1
$env:PYTHONPATH='src'; c:/Users/Milos/PythonProjects/satellite-land-classification-dl/.venv/Scripts/python.exe run.py --run-baseline --config configs/efficientnet_b0.stage2.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits --reports-output artifacts/reports/efficientnet_b0_stage2_final.json --checkpoints-output checkpoints/efficientnet_b0/stage2
```

CLI now supports overriding config values at runtime (for notebooks and experiment sweeps) without creating extra config files.

```powershell
$env:PYTHONPATH='src'; c:/Users/Milos/PythonProjects/satellite-land-classification-dl/.venv/Scripts/python.exe run.py --run-baseline --config configs/experiment.defaults.json --model-name efficientnet_b0 --dataset-root data/EuroSAT --seed 42 --epochs 30 --batch-size 16 --learning-rate 0.0001 --augmentation-mode flips --use-pretrained --freeze-backbone --splits-output artifacts/splits --reports-output artifacts/reports/override_example.json --checkpoints-output checkpoints/override_example
```

## Quality Gates (Phase 7)

Run these commands from the repository root in an activated virtual environment.

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest -q tests
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m ruff check src tests
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m mypy src
```

## Active Components

| Component Type | Active Files |
| --- | --- |
| Configs | `configs/baseline_cnn.json`, `configs/efficientnet_b0.stage1.json`, `configs/efficientnet_b0.stage2.json`, `configs/experiment.defaults.json` |
| Notebooks | `notebooks/eurosat_baseline_kaggle.ipynb`, `notebooks/eurosat_efficientnet_kaggle.ipynb` |
| Reports | `artifacts/reports/baseline_cnn.json`, `artifacts/reports/efficientnet_b0_stage1_final.json`, `artifacts/reports/efficientnet_b0_stage2_final.json`, `results/final/efficientnet_b0_final.json` |

## Reproducibility Assets (Phase 1)

- Canonical defaults: `configs/experiment.defaults.json`
- Locked split manifest: `artifacts/splits/split_manifest.json`
- Evaluation protocol and acceptance criteria: `docs/evaluation_protocol.md`
- Deterministic split artifacts (generated):
  - `artifacts/splits/train_split.json`
  - `artifacts/splits/validation_split.json`
  - `artifacts/splits/test_split.json`
  - `artifacts/splits/split_summary.json`

## Notes

- Baseline CNN training and evaluation are implemented end-to-end with checkpoint and report outputs.
- Training reproducibility is enforced through fixed split seed plumbing and global seed setup.
- The current project specification targets the RGB JPEG dataset in `data/EuroSAT`, not the multispectral TIFF variant.
- ResNet50-specific implementation should remain Bojan's responsibility unless explicitly requested otherwise.
