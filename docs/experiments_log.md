# Experiments Log

This log tracks baseline and EfficientNetB0 runs for the EuroSAT project.

| Date | Model | Stage | Epochs | Accuracy | F1 |
|---|---|---|---:|---:|---:|
| 2026-03-12 | efficientnet_b0 | stage1_smoke | 1 | 0.8973 | 0.8929 |
| 2026-03-12 | efficientnet_b0 | stage2_smoke | 1 | 0.9188 | 0.9190 |
| YYYY-MM-DD | efficientnet_b0 | stage1_full | TBD | TBD | TBD |
| YYYY-MM-DD | efficientnet_b0 | stage2_full | TBD | TBD | TBD |
| YYYY-MM-DD | baseline_cnn | baseline_reference | TBD | TBD | TBD |

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

## Thesis-Ready Reporting Tables (Fill After Today's Rerun)

Use only artifacts from the new rerun session. Keep historical values above as reference only.

### 1. Ablation Table

| Model Family | Run ID | Stage | Augmentation | Learning Rate | Seed | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| baseline_cnn | TBD | baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| baseline_cnn | TBD | baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| baseline_cnn | TBD | baseline | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| efficientnet_b0 | TBD | stage1 | TBD | TBD | TBD | TBD | TBD | TBD | no |
| efficientnet_b0 | TBD | stage2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| efficientnet_b0 | TBD | stage2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| efficientnet_b0 | TBD | stage2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 2. Model Selection Table

| Model Family | Selection Pool Definition | Selection Metric | Selected Run ID | Selected Val Macro F1 | Notes |
|---|---|---|---|---:|---|
| baseline_cnn | all baseline runs | val_f1_best | TBD | TBD | test metrics are holdout-only |
| efficientnet_b0 | stage2 runs only | val_f1_best | TBD | TBD | stage1 is transfer setup, not final comparator |

### 3. Final Holdout Test Table

| Model Family | Selected Run ID | Seed | Test Accuracy | Test Macro F1 | Confusion Matrix Source | Per-Class Metrics Source |
|---|---|---:|---:|---:|---|---|
| baseline_cnn | TBD | TBD | TBD | TBD | TBD | TBD |
| efficientnet_b0 | TBD | TBD | TBD | TBD | TBD | TBD |

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
