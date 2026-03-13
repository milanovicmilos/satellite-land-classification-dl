# EfficientNetB0 Revision Change Log (Kaggle vs Local)

## Project Context
This project is EuroSAT land-use classification in Python with stratified 70/15/15 split, fixed seed 42, and evaluation by accuracy, macro F1-score, confusion matrix, precision, and recall.

## Scope Of This Revision
- Differential analysis of poor Kaggle full runs versus current local full runs.
- Standardization of augmentation behavior (`none | flips | full`).
- Results hygiene to remove legacy report noise.
- Consolidation of final EfficientNetB0 source-of-truth artifacts.

## What Changed
1. Added configurable `augmentation_mode` in shared training config parsing and dataloader construction.
2. Implemented augmentation routing in `SplitJsonLoaderFactory` with explicit validation for `none`, `flips`, and `full`.
3. Strengthened reproducibility by enabling deterministic PyTorch algorithms in seed setup.
4. Archived legacy/smoke reports and temporary training logs from active reports folder.
5. Generated consolidated final artifact: `results/final/efficientnet_b0_final_full.json`.

## Differential Analysis Summary
### Kaggle Poor Stage 2 (from archived Kaggle full run)
- `early_stopping_patience`: 5
- `learning_rate`: 0.0003
- `scheduler_patience`: 2
- `early_stopping_min_delta`: 0.0005
- Effective train augmentation previously included heavier transforms in full mode.
- Result: test accuracy ~0.9180, macro F1 ~0.9191.

### Local Final Stage 2 (current source of truth)
- `augmentation_mode`: `flips`
- `early_stopping_patience`: 2
- `learning_rate`: 0.0001
- `scheduler_patience`: 1
- `early_stopping_min_delta`: 0.001
- Result: test accuracy 0.9696, macro F1 0.9690.

### Primary Error In Previous Runs
The major issue was not one parameter in isolation; it was the interaction of:
- overly permissive Stage 2 patience,
- higher Stage 2 learning rate,
- and stronger augmentation noise on 64x64 inputs.

This delayed stopping and increased instability after peak validation quality.

## Why Local Results Are Superior
1. Reduced optimization noise in Stage 2 with lower LR (`0.0001`).
2. Faster overfit protection from tighter early stopping (`patience=2`).
3. Lighter augmentation regime (`flips`) better aligned with 64x64 EuroSAT image scale.
4. Cleaner experiment hygiene and reproducibility controls.

## Hygiene Actions Performed
Active reports directory now keeps only:
- `artifacts/reports/baseline_cnn_full.json`
- `artifacts/reports/efficientnet_b0_stage1_final.json`
- `artifacts/reports/efficientnet_b0_stage2_final.json`

All older smoke/temporary/legacy reports were moved into:
- `artifacts/reports/archive/`

## Reproducibility Notes
- Seed flow is fixed at `42` through split configuration and orchestrator seeding.
- `set_seed` controls Python `random`, NumPy, and PyTorch (CPU/CUDA), and now requests deterministic algorithms.
- Residual caveat: exact cross-hardware bitwise equality can still vary by driver/CUDA stack and nondeterministic ops reported in warning mode.

## Team Boundary Note
This revision only updates shared infrastructure and EfficientNetB0 path (Miloš scope), while preserving ResNet50 ownership boundary for Bojan.

