# Experiments Log

This log tracks baseline and EfficientNetB0 runs for the EuroSAT project.

| Date | Model | Stage | Epochs | Accuracy | F1 |
|---|---|---|---:|---:|---:|
| 2026-03-12 | efficientnet_b0 | stage1_smoke | 1 | 0.8973 | 0.8929 |
| 2026-03-12 | efficientnet_b0 | stage2_smoke | 1 | 0.9188 | 0.9190 |
| 2026-04-10 | baseline_cnn | baseline_grid_best | 50 | 0.961975 | 0.960863 |
| 2026-04-10 | efficientnet_b0 | stage1_reference | 9 | 0.926173 | 0.923145 |
| 2026-04-10 | efficientnet_b0 | stage2_reference_best | 24 | 0.974321 | 0.973621 |

## Notes
- Use the same split seed and split artifacts when comparing model families.
- Prefer macro F1 as the primary comparison metric when class-level balance matters.

## Historical Reference Results (2026-04-10, pre-thesis protocol lock)

All runs below used the same split protocol (70/15/15, stratified, seed 42).

### Baseline CNN Grid

| Run ID | Augmentation | Learning Rate | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|---:|
| baseline_reference_none | none | 0.00100 | 0.912346 | 0.908949 |
| baseline_flips | flips | 0.00100 | 0.947901 | 0.946238 |
| baseline_flips_low_lr | flips | 0.00050 | 0.961975 | 0.960863 |

Selected baseline for reporting (by validation macro F1): baseline_flips_low_lr.

### EfficientNetB0 Grid

| Run ID | Stage | Augmentation | Learning Rate | Test Accuracy | Test Macro F1 |
|---|---|---|---:|---:|---:|
| efficientnet_stage1_reference | stage1 | flips | 0.00100 | 0.926173 | 0.923145 |
| efficientnet_stage2_reference | stage2 | flips | 0.00010 | 0.974321 | 0.973621 |
| efficientnet_stage2_low_lr | stage2 | flips | 0.00005 | 0.971358 | 0.970286 |
| efficientnet_stage2_no_aug | stage2 | none | 0.00010 | 0.966914 | 0.965938 |

Selected EfficientNetB0 for reporting (by validation macro F1 within stage2 pool): efficientnet_stage2_reference.

### Historical vs Current Comparison

| Model Scenario | Previous Accuracy | Current Accuracy | Delta Accuracy | Previous Macro F1 | Current Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_cnn reference | 0.907901 | 0.912346 | +0.004445 | 0.904257 | 0.908949 | +0.004692 |
| baseline_cnn best | 0.907901 | 0.961975 | +0.054074 | 0.904257 | 0.960863 | +0.056606 |
| efficientnet_b0 stage1 | 0.915556 | 0.926173 | +0.010617 | 0.911969 | 0.923145 | +0.011175 |
| efficientnet_b0 stage2 | 0.973333 | 0.974321 | +0.000988 | 0.972363 | 0.973621 | +0.001258 |

### Methodology Warning (Important)

- Do not choose the final model by test metrics when comparing many runs.
- Correct protocol: tune and select on validation metric, then evaluate once on the locked test split.
- Current table is useful for transparency and debugging, but final thesis claims should clearly state the model selection rule.

## Thesis-Ready Reporting Tables (Filled From Kaggle Execution Snapshots)

Source snapshots:
- `results/eurosat-baseline.ipynb`
- `results/eurosat-efficientnet.ipynb`

Papermill execution start timestamps in these snapshots:
- Baseline notebook: 2026-04-10T17:51:31
- EfficientNet notebook: 2026-04-10T17:51:51

### 1. Ablation Table

| Model Family | Run ID | Stage | Augmentation | Learning Rate | Seed | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| baseline_cnn | baseline_reference_none | baseline | none | 0.00100 | 42 | 0.915363 | 0.912346 | 0.908949 | no |
| baseline_cnn | baseline_flips | baseline | flips | 0.00100 | 42 | 0.951813 | 0.947901 | 0.946238 | no |
| baseline_cnn | baseline_flips_low_lr | baseline | flips | 0.00050 | 42 | 0.959781 | 0.961975 | 0.960863 | yes |
| efficientnet_b0 | efficientnet_stage1_reference | stage1 | flips | 0.00100 | 42 | 0.920765 | 0.926173 | 0.923145 | no |
| efficientnet_b0 | efficientnet_stage2_reference | stage2 | flips | 0.00010 | 42 | 0.978875 | 0.974321 | 0.973621 | yes |
| efficientnet_b0 | efficientnet_stage2_low_lr | stage2 | flips | 0.00005 | 42 | 0.975499 | 0.971358 | 0.970286 | no |
| efficientnet_b0 | efficientnet_stage2_no_aug | stage2 | none | 0.00010 | 42 | 0.965855 | 0.966914 | 0.965938 | no |

### 2. Model Selection Table

| Model Family | Selection Pool Definition | Selection Metric | Selected Run ID | Selected Val Macro F1 | Notes |
|---|---|---|---|---:|---|
| baseline_cnn | all baseline runs | val_f1_best | baseline_flips_low_lr | 0.959781 | test metrics are holdout-only |
| efficientnet_b0 | stage2 runs only | val_f1_best | efficientnet_stage2_reference | 0.978875 | stage1 is transfer setup, not final comparator |

### 3. Final Holdout Test Table

| Model Family | Selected Run ID | Seed | Test Accuracy | Test Macro F1 | Confusion Matrix Source | Per-Class Metrics Source |
|---|---|---:|---:|---:|---|---|
| baseline_cnn | baseline_flips_low_lr | 42 | 0.961975 | 0.960863 | /kaggle/working/artifacts/reports/baseline/baseline_flips_low_lr.json | /kaggle/working/artifacts/reports/baseline/baseline_flips_low_lr.json |
| efficientnet_b0 | efficientnet_stage2_reference | 42 | 0.974321 | 0.973621 | /kaggle/working/artifacts/reports/efficientnet/efficientnet_stage2_reference.json | /kaggle/working/artifacts/reports/efficientnet/efficientnet_stage2_reference.json |

### 4. Snapshot Interpretation Summary

- Baseline augmentation effect: `baseline_flips` improves macro F1 by +0.037289 over `baseline_reference_none`.
- Baseline learning-rate effect: `baseline_flips_low_lr` improves macro F1 by +0.014625 over `baseline_flips`.
- EfficientNet staged fine-tuning effect: Stage 2 reference improves macro F1 by +0.050476 over Stage 1 reference.
- EfficientNet Stage 2 ablation: `stage2_low_lr` is -0.003335 macro F1 vs `stage2_reference`; `stage2_no_aug` is -0.007683.
- Final model-family comparison: selected EfficientNet run is +0.012758 macro F1 and +0.012346 accuracy over selected baseline run.
- Validation-to-test gap for selected runs: baseline +0.001082 macro F1; EfficientNet -0.005254 macro F1.

## Run ID Glossary

| Term | Meaning |
|---|---|
| reference | Canonical setup used as a comparison anchor. |
| low_lr | Same setup with a lower learning rate. |
| no_aug | Data augmentation disabled. |
| stage1 | Frozen-backbone transfer phase. |
| stage2 | Unfrozen fine-tuning phase resumed from stage1 checkpoint. |

## Scientific Reporting Checklist

- Keep split protocol fixed (70/15/15, stratified, same seed) for all runs compared in one table.
- Select models using validation metric only (`val_f1_best`), not test metrics.
- Report test metrics only for the selected run(s) as final holdout performance.
- If running multiple seeds, report mean and standard deviation in addition to per-seed rows.

## Engineering Quality Review Notes

- Validation-driven model selection is correctly implemented in both notebooks and manifests.
- Selected runs are consistent with highest validation macro F1 in their respective selection pools.
- PyTorch determinism warning appears in the snapshots because `CUBLAS_WORKSPACE_CONFIG` is not set in the Kaggle process environment.
- `pip install -e .` shows resolver conflict warnings with preinstalled Kaggle packages; runs still completed successfully, but environment isolation should be improved for strict reproducibility claims.
