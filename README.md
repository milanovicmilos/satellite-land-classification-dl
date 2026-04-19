# EuroSAT Land-Use Classification

Deep learning project for EuroSAT RGB land-use classification (10 classes), with three model families:

- baseline CNN (from scratch)
- EfficientNetB0 fine-tuning
- ResNet50 fine-tuning

The project emphasizes reproducibility (fixed seed + deterministic stratified split), consistent evaluation across all model families, and clean architecture.

## Overview

- Dataset: EuroSAT RGB
- Task: multi-class image classification (10 labels)
- Split policy: stratified `70/15/15` (train/validation/test)
- Evaluation metrics: accuracy, macro F1-score, confusion matrix, per-class precision, per-class recall

This repository includes both local CLI execution and Kaggle-oriented experiment flows.

## Repository Structure

```text
src/eurosat_classifier/
	domain/          Entities and metric contracts
	application/     Use cases and orchestration
	infrastructure/  Data access, models, training, evaluation, checkpointing
	entrypoints/     CLI entrypoint
tests/             Unit and smoke tests
configs/           Experiment configurations
notebooks/         Training notebooks (Kaggle-oriented)
results/           Exported experiment outputs and snapshots
docs/              Final report and project documents
```

## Quick Start

### Prerequisites

- Python `3.12`
- `pip`
- EuroSAT RGB dataset available locally at `data/EuroSAT/`

See dataset notes in `data/README.md`.

### Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Install development dependencies (tests, lint, type checks):

```powershell
pip install -e ".[dev]"
```

## Run The Pipeline

Run commands from repository root.

Validate config and wiring:

```powershell
$env:PYTHONPATH='src'; python -m eurosat_classifier --dry-run --config configs/baseline_cnn.json
```

Generate deterministic split artifacts:

```powershell
$env:PYTHONPATH='src'; python -m eurosat_classifier --prepare-dataset --config configs/baseline_cnn.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits
```

Train and evaluate baseline CNN:

```powershell
$env:PYTHONPATH='src'; python -m eurosat_classifier --run-baseline --config configs/baseline_cnn.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits --reports-output artifacts/reports/baseline_metrics.json --checkpoints-output checkpoints/baseline
```

## Kaggle Workflow

Kaggle is a first-class execution target in this project.

Primary Kaggle runner notebooks:

- `notebooks/eurosat_baseline_kaggle.ipynb`
- `notebooks/eurosat_efficientnet_kaggle.ipynb`
- `notebooks/eurosat_resnet50_kaggle.ipynb`

Result snapshot notebooks:

- `results/eurosat-baseline.ipynb`
- `results/eurosat-efficientnet.ipynb`
- `results/eurosat-resnet.ipynb`

Typical Kaggle runtime layout used by the notebooks:

- Dataset input: `/kaggle/input/.../EuroSAT`
- Generated splits: `/kaggle/working/artifacts/splits`
- Reports: `/kaggle/working/artifacts/reports/...`
- Checkpoints: `/kaggle/working/checkpoints/...`

The notebooks package outputs into zip files in `/kaggle/working` so they can be downloaded and archived.

## Quality Checks

```powershell
$env:PYTHONPATH='src'; python -m pytest -q tests
$env:PYTHONPATH='src'; python -m ruff check src tests
$env:PYTHONPATH='src'; python -m mypy src
```

## Configurations

- `configs/experiment.defaults.json`
- `configs/baseline_cnn.json`
- `configs/efficientnet_b0.stage1.json`
- `configs/efficientnet_b0.stage2.json`
- `configs/resnet50.stage1.json`
- `configs/resnet50.stage2.json`
- `configs/resnet50.template.json`

## Project Paper

- Final PDF: `docs/EuroSAT_klasifikacija_Milanovic_Zivanic.pdf`
- Working document: `docs/EuroSAT_klasifikacija_Milanovic_Zivanic.docx`
- Original specification: `docs/specifikacija.txt`

## Team

This project was developed collaboratively by both team members.

- Miloš Milanović: EfficientNetB0
- Bojan Živanić: ResNet50
