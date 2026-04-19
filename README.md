# EuroSAT Land-Use Classification

Deep learning project for EuroSAT RGB land-use classification (10 classes), with three model families:

- baseline CNN (from scratch)
- EfficientNetB0 fine-tuning
- ResNet50 fine-tuning

The project emphasizes reproducibility (fixed seed + deterministic stratified split), consistent evaluation across all model families, and clean architecture.

## Problem Context

Accurate land-use classification from satellite imagery is important for urban planning, agriculture monitoring, environmental analysis, and change detection. EuroSAT is a common benchmark for this task, but model comparisons are only meaningful when data splits and evaluation criteria are strictly controlled.

This project is built around that idea: not only training models, but comparing them fairly and reproducibly.

## Project Goal

The goal is to compare a baseline CNN against two transfer-learning approaches (EfficientNetB0 and ResNet50) on the same EuroSAT setup, then analyze trade-offs between performance and training strategy.

Key decisions:

- one shared deterministic split for all model families
- one shared evaluation protocol and metric set
- model comparison based on holdout test results

## Overview

- Dataset: EuroSAT RGB
- Task: multi-class image classification (10 labels)
- Split policy: stratified `70/15/15` (train/validation/test)
- Evaluation metrics: accuracy, macro F1-score, confusion matrix, per-class precision, per-class recall

This repository includes both local CLI execution and Kaggle-oriented experiment flows.

## Methodology Flow

1. Build a deterministic stratified split (`70/15/15`) from EuroSAT RGB.
2. Train each model family using the same split artifacts.
3. Select candidate runs using validation performance.
4. Evaluate selected runs on the holdout test split.
5. Compare models using the same metrics and exported reports.

## Final Results (Holdout Test)

The table below summarizes the selected final runs from the stored Kaggle reports.

| Model | Final Run | Test Accuracy | Test Macro F1 | Notes |
| --- | --- | --- | --- | --- |
| Baseline CNN | `baseline_flips_low_lr` | 0.96198 | 0.96086 | reference model trained from scratch |
| EfficientNetB0 | `efficientnet_stage2_reference` | 0.97432 | 0.97362 | two-stage fine-tuning |
| ResNet50 | `resnet50_stage2_reference` | 0.97877 | 0.97847 | two-stage fine-tuning |

Result sources:

- `results/baseline/kaggle/working/artifacts/reports/baseline/baseline_holdout_report.csv`
- `results/efficientnet/kaggle/working/artifacts/reports/efficientnet/efficientnet_holdout_report.csv`
- `results/resnet/kaggle/working/artifacts/reports/resnet50/resnet50_holdout_report.csv`

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

## Reproducibility Checklist

Use this sequence to reproduce experiments consistently:

1. Install dependencies in a fresh Python 3.12 environment.
2. Keep the same split seed (`42`) and split ratios (`0.7/0.15/0.15`).
3. Generate split artifacts once using `--prepare-dataset`.
4. Run model experiments using the same split artifact paths.
5. Select runs by validation macro F1, then compare holdout test metrics.

Recommended configs:

- Baseline: `configs/baseline_cnn.json`
- EfficientNetB0 stage1: `configs/efficientnet_b0.stage1.json`
- EfficientNetB0 stage2: `configs/efficientnet_b0.stage2.json`
- ResNet50 stage1: `configs/resnet50.stage1.json`
- ResNet50 stage2: `configs/resnet50.stage2.json`

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

## License

This project is licensed under the MIT License. See `LICENSE` for details.
