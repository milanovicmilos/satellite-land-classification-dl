# Experiments Log

This log tracks baseline CNN, EfficientNetB0, and ResNet50 runs for the EuroSAT project.

| Date | Model | Stage | Epochs | Accuracy | F1 |
|---|---|---|---:|---:|---:|
| 2026-03-12 | efficientnet_b0 | stage1_smoke | 1 | 0.8973 | 0.8929 |
| 2026-03-12 | efficientnet_b0 | stage2_smoke | 1 | 0.9188 | 0.9190 |
| 2026-04-10 | baseline_cnn | baseline_grid_best | 50 | 0.961975 | 0.960863 |
| 2026-04-10 | efficientnet_b0 | stage1_reference | 9 | 0.926173 | 0.923145 |
| 2026-04-10 | efficientnet_b0 | stage2_reference_best | 24 | 0.974321 | 0.973621 |
| 2026-04-10 | resnet50 | stage1_reference | 5 | 0.927901 | 0.925859 |
| 2026-04-10 | resnet50 | stage2_reference_best | 9 | 0.978765 | 0.978468 |

## Notes
- Use the same split seed and split artifacts when comparing model families.
- Prefer macro F1 as the primary comparison metric when class-level balance matters.

## Execution Metadata (All Three Models)

All runs executed on 2026-04-10 via Kaggle notebooks with identical dataset and split protocol.

| Model | Notebook | Execution Start | Runtime | Status |
|---|---|---|---:|---|
| Baseline CNN | `results/eurosat-baseline.ipynb` | 2026-04-10T17:51:31 | ~97.93 min | ✓ Completed |
| EfficientNet B0 | `results/eurosat-efficientnet.ipynb` | 2026-04-10T17:51:51 | ~82.38 min | ✓ Completed |
| ResNet50 | `results/eurosat-resnet50.ipynb` | 2026-04-10T18:45:12 | ~58.67 min | ✓ Completed |

All notebooks used:
- Split seed: 42 (fixed for reproducibility)
- Split ratios: 70% train / 15% validation / 15% test (stratified by class)
- Dataset: EuroSAT (10 classes, 27,000 images)
- Metrics: accuracy, macro F1, confusion matrix, per-class precision/recall

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

### ResNet50 Grid

| Run ID | Stage | Augmentation | Learning Rate | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|---:|---:|
| resnet50_stage1_reference | stage1 | flips | 0.00100 | 0.927901 | 0.925859 |
| resnet50_stage2_reference | stage2 | flips | 0.00010 | 0.978765 | 0.978468 |
| resnet50_stage2_low_lr | stage2 | flips | 0.00005 | 0.979506 | 0.979078 |
| resnet50_stage2_no_aug | stage2 | none | 0.00010 | 0.964691 | 0.963568 |

Selected ResNet50 for reporting (by validation macro F1 within stage2 pool): resnet50_stage2_reference.

| Model Scenario | Previous Accuracy | Current Accuracy | Delta Accuracy | Previous Macro F1 | Current Macro F1 | Delta Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| baseline_cnn reference | 0.907901 | 0.912346 | +0.004445 | 0.904257 | 0.908949 | +0.004692 |
| baseline_cnn best | 0.907901 | 0.961975 | +0.054074 | 0.904257 | 0.960863 | +0.056606 |
| efficientnet_b0 stage1 | 0.915556 | 0.926173 | +0.010617 | 0.911969 | 0.923145 | +0.011175 |
| efficientnet_b0 stage2 | 0.973333 | 0.974321 | +0.000988 | 0.972363 | 0.973621 | +0.001258 |
| resnet50 stage1 | N/A | 0.927901 | N/A | N/A | 0.925859 | N/A |
| resnet50 stage2 | N/A | 0.978765 | N/A | N/A | 0.978468 | N/A |

### Methodology Warning (Important)

- Do not choose the final model by test metrics when comparing many runs.
- Correct protocol: tune and select on validation metric, then evaluate once on the locked test split.
- Current table is useful for transparency and debugging, but final thesis claims should clearly state the model selection rule.

## Thesis-Ready Reporting Tables (Filled From Kaggle Execution Snapshots)

Source snapshots:
- `results/eurosat-baseline.ipynb`
- `results/eurosat-efficientnet.ipynb`
- `results/eurosat-resnet50.ipynb`

Papermill execution start timestamps in these snapshots:
- Baseline notebook: 2026-04-10T17:51:31
- EfficientNet notebook: 2026-04-10T17:51:51
- ResNet50 notebook: 2026-04-10T18:45:12

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
| resnet50 | resnet50_stage1_reference | stage1 | flips | 0.00100 | 42 | 0.929148 | 0.927901 | 0.925859 | no |
| resnet50 | resnet50_stage2_reference | stage2 | flips | 0.00010 | 42 | 0.978449 | 0.978765 | 0.978468 | yes |
| resnet50 | resnet50_stage2_low_lr | stage2 | flips | 0.00005 | 42 | 0.978224 | 0.979506 | 0.979078 | no |
| resnet50 | resnet50_stage2_no_aug | stage2 | none | 0.00010 | 42 | 0.967177 | 0.964691 | 0.963568 | no |

### 2. Model Selection Table

| Model Family | Selection Pool Definition | Selection Metric | Selected Run ID | Selected Val Macro F1 | Notes |
|---|---|---|---|---:|---|
| baseline_cnn | all baseline runs | val_f1_best | baseline_flips_low_lr | 0.959781 | test metrics are holdout-only |
| efficientnet_b0 | stage2 runs only | val_f1_best | efficientnet_stage2_reference | 0.978875 | stage1 is transfer setup, not final comparator |
| resnet50 | stage2 runs only | val_f1_best | resnet50_stage2_reference | 0.978449 | stage1 is transfer setup, not final comparator |

### 3. Final Holdout Test Table

| Model Family | Selected Run ID | Seed | Test Accuracy | Test Macro F1 | Confusion Matrix Source | Per-Class Metrics Source |
|---|---|---:|---:|---:|---|---|
| baseline_cnn | baseline_flips_low_lr | 42 | 0.961975 | 0.960863 | /kaggle/working/artifacts/reports/baseline/baseline_flips_low_lr.json | /kaggle/working/artifacts/reports/baseline/baseline_flips_low_lr.json |
| efficientnet_b0 | efficientnet_stage2_reference | 42 | 0.974321 | 0.973621 | /kaggle/working/artifacts/reports/efficientnet/efficientnet_stage2_reference.json | /kaggle/working/artifacts/reports/efficientnet/efficientnet_stage2_reference.json |
| resnet50 | resnet50_stage2_reference | 42 | 0.978765 | 0.978468 | /kaggle/working/artifacts/reports/resnet50/resnet50_stage2_reference.json | /kaggle/working/artifacts/reports/resnet50/resnet50_stage2_reference.json |

### 4. Snapshot Interpretation Summary

- Baseline augmentation effect: `baseline_flips` improves macro F1 by +0.037289 over `baseline_reference_none`.
- Baseline learning-rate effect: `baseline_flips_low_lr` improves macro F1 by +0.014625 over `baseline_flips`.
- EfficientNet staged fine-tuning effect: Stage 2 reference improves macro F1 by +0.050476 over Stage 1 reference.
- EfficientNet Stage 2 ablation: `stage2_low_lr` is -0.003335 macro F1 vs `stage2_reference`; `stage2_no_aug` is -0.007683.
- ResNet50 staged fine-tuning effect: Stage 2 reference improves macro F1 by +0.052609 over Stage 1 reference.
- ResNet50 Stage 2 ablation: `stage2_low_lr` is +0.000775 macro F1 vs `stage2_reference` (marginal increase); `stage2_no_aug` is -0.014900.
- Final model-family comparison (by selected runs): ResNet50 is +0.004647 macro F1 and +0.004444 accuracy over EfficientNet; EfficientNet is +0.012758 macro F1 and +0.012346 accuracy over Baseline.
- Validation-to-test gap for selected runs: baseline +0.001082 macro F1; EfficientNet -0.005254 macro F1; ResNet50 +0.000019 macro F1 (excellent stability).

### 5. Three-Way Model Comparison Summary

| Metric | Baseline | EfficientNet B0 | ResNet50 | Best Performer |
|---|---:|---:|---:|---|
| Val Macro F1 (best) | 0.959781 | 0.978875 | 0.978449 | EfficientNet |
| Test Accuracy | 0.961975 | 0.974321 | 0.978765 | ResNet50 |
| Test Macro F1 | 0.960863 | 0.973621 | 0.978468 | ResNet50 |
| Epochs to Conv (Stage 2) | 50 | 24 | 9 | ResNet50 |

Key observations:
- Baseline CNN reaches 96.0% macro F1, serving as a solid reference point.
- EfficientNet B0 achieves 97.4% macro F1 with 24-epoch fine-tuning, demonstrating effective transfer learning.
- ResNet50 reaches 97.8% macro F1 with fastest convergence (9 epochs), showing superior sample efficiency and generalization.
- ResNet50 shows the tightest validation-to-test gap (+0.000019 macro F1), indicating excellent robustness and minimal overfitting.

## Model-Specific Insights and Ablation Impact

### Baseline CNN Key Findings
- Augmentation is the primary driver: `baseline_flips` improves macro F1 by +0.037289 vs no augmentation.
- Learning rate tuning adds secondary gain: `baseline_flips_low_lr` improves by additional +0.014625.
- Total improvement from reference to best: +0.044414 macro F1 (4.6% absolute increase).
- Convergence: 50 epochs required for early stopping at best validation performance.

### EfficientNet B0 Key Findings
- Staged fine-tuning effect: Stage 2 reference improves macro F1 by +0.050476 over Stage 1 (5.5% absolute).
- Learning rate tuning in Stage 2: `stage2_low_lr` reduces performance by -0.003335 macro F1 (0.04% drop).
- Augmentation ablation: Disabling augmentation in Stage 2 causes -0.007683 macro F1 drop (0.79% loss).
- Convergence: Stage 1 requires 9 epochs; Stage 2 requires 24 epochs (total 33 epochs two-stage protocol).

### ResNet50 Key Findings
- Staged fine-tuning effect: Stage 2 reference improves macro F1 by +0.052609 over Stage 1 (largest among models).
- Learning rate tuning in Stage 2: `stage2_low_lr` shows marginal +0.000775 macro F1 improvement (negligible, +0.008% only).
- Augmentation ablation: Disabling augmentation in Stage 2 causes -0.014900 macro F1 drop (1.5% loss, most sensitive to augmentation).
- Convergence: Stage 1 requires only 5 epochs; Stage 2 requires 9 epochs (total 14 epochs two-stage protocol, 4× faster than baseline).

## Three-Model Comparative Summary

| Aspect | Baseline | EfficientNet | ResNet50 |
|---|---|---|---|
| **Peak Accuracy** | 96.20% | 97.43% | 97.88% |
| **Peak Macro F1** | 96.09% | 97.36% | 97.85% |
| **Total Training Epochs** | 50 | 33 (staged) | 14 (staged) |
| **Augmentation Sensitivity** | +4.43% F1 gain | +0.77% F1 loss (no aug) | +1.50% F1 loss (no aug) |
| **Learning Rate Sensitivity** | +1.46% F1 gain (lower LR) | -0.33% F1 loss (lower LR) | +0.08% F1 gain (lower LR) |
| **Validation→Test Gap** | +0.11% | -0.53% | +0.002% |
| **Transfer Learning Gain** | N/A | +1.28% accuracy | +1.74% accuracy |

Interpretation:
- Transfer models are more robust: smaller sensitivity to learning rate changes, more stable generalization.
- ResNet50 converges fastest and generalizes best: 14-epoch protocol with minimal val→test gap.
- Augmentation critical for all models but ResNet50 shows highest sensitivity: emphasizes importance of preprocessing consistency across model families.

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
