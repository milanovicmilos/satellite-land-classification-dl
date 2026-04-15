# Evaluation Protocol

## Purpose

Define canonical experiment and reporting rules so baseline CNN, EfficientNetB0, and ResNet50 are compared under identical conditions.

## Dataset Variant

- Dataset path: `data/EuroSAT`
- Input format: RGB JPEG
- Class count: 10

## Canonical Split

- Strategy: stratified by class
- Train ratio: 0.70
- Validation ratio: 0.15
- Test ratio: 0.15
- Fixed seed: 42

Split metadata must be tracked via `artifacts/splits/split_manifest.json` and reused by all model families.

## Required Metrics

- Primary metric: accuracy
- Secondary metrics:
  - macro F1-score
  - confusion matrix
  - per-class precision
  - per-class recall

## Model Comparison Table Schema

Schema id: `eurosat-model-comparison-v1`

Required columns:
- `model_name`
- `split_seed`
- `train_ratio`
- `validation_ratio`
- `test_ratio`
- `accuracy`
- `macro_f1_score`
- `checkpoint_path`
- `run_id`
- `created_at_utc`

## Acceptance Criteria

All three model families follow identical evaluation setup and acceptance rules:

### Baseline CNN

- End-to-end train/validation/test execution completes.
- Required metrics are produced for test split.
- Confusion matrix artifact is exported.
- Output row follows `eurosat-model-comparison-v1` schema.
- Selection by validation macro F1 across all baseline runs.

### EfficientNetB0 (Milos Scope)

- Stage 1 (frozen backbone) execution completes.
- Stage 2 (fine-tuning/unfreeze) execution completes.
- Required metrics are produced for test split.
- Selection by validation macro F1 within Stage 2 pool only.
- Stage 1 serves as a transfer baseline, not a final comparator.
- Comparison against baseline CNN and ResNet50 is provided in summary reports.

### ResNet50 (Bojan Scope)

- Stage 1 (frozen backbone) execution completes.
- Stage 2 (fine-tuning/unfreeze) execution completes.
- Required metrics are produced for test split.
- Selection by validation macro F1 within Stage 2 pool only.
- Stage 1 serves as a transfer baseline, not a final comparator.
- Integration follows the same split, protocol, and table schema as baseline and EfficientNet.
- Ownership boundary is preserved according to project instructions.
