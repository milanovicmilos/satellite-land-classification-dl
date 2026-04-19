# EuroSAT Land-Use Classification

Python project for 10-class land-use classification on EuroSAT RGB images.

This repository is organized to fully satisfy the required project deliverables:

1. Source code
2. Documentation with setup and run instructions
3. Final paper in PDF format

## 1. Source Code

Core implementation is in:

- `src/eurosat_classifier/`
- `tests/`
- `configs/`
- `run.py`
- `pyproject.toml`

Main architecture follows layered boundaries:

- `domain`: entities and metrics contracts
- `application`: use cases and orchestration services
- `infrastructure`: dataset loading, model wiring, evaluation, checkpointing
- `entrypoints`: command-line interface

## 2. Documentation And How To Run

### Prerequisites

- Python 3.12
- pip
- EuroSAT RGB dataset available locally under `data/EuroSAT/`

Dataset notes are in `data/README.md`.

### Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Optional development tools:

```powershell
pip install -e ".[dev]"
```

### Basic CLI Usage

Use commands from repository root.

Dry run:

```powershell
$env:PYTHONPATH='src'; python -m eurosat_classifier --dry-run --config configs/baseline_cnn.json
```

Prepare deterministic dataset split artifacts:

```powershell
$env:PYTHONPATH='src'; python -m eurosat_classifier --prepare-dataset --config configs/baseline_cnn.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits
```

Run baseline training and evaluation:

```powershell
$env:PYTHONPATH='src'; python -m eurosat_classifier --run-baseline --config configs/baseline_cnn.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits --reports-output artifacts/reports/baseline_metrics.json --checkpoints-output checkpoints/baseline
```

Run quality checks:

```powershell
$env:PYTHONPATH='src'; python -m pytest -q tests
$env:PYTHONPATH='src'; python -m ruff check src tests
$env:PYTHONPATH='src'; python -m mypy src
```

### Useful Project Files

- `configs/experiment.defaults.json`
- `configs/baseline_cnn.json`
- `configs/efficientnet_b0.stage1.json`
- `configs/efficientnet_b0.stage2.json`
- `configs/resnet50.stage1.json`
- `configs/resnet50.stage2.json`
- `results/` (experiment outputs and snapshots)
- `notebooks/` (Kaggle-oriented notebooks)

## 3. Final PDF Version Of The Paper

Final paper is included at:

- `docs/EuroSAT_klasifikacija_Milanovic_Zivanic.pdf`

## Team

- Miloš Milanović: EfficientNetB0 and shared infrastructure
- Bojan Živanić: ResNet50
