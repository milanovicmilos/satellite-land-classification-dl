# EuroSAT Experiment Report (Working Source for Thesis Writing)

## 1. Document Role

This document is a consolidated source of verified experiment information for writing the final thesis report.

Scope of this version:
- Included: shared Baseline CNN results and EfficientNetB0 results (Milos scope).
- Pending: ResNet50 results (Bojan scope), with a ready-to-fill template.

This file is meant to reduce ambiguity when converting raw notebook outputs into thesis text.

## 2. Result Sources and Execution Metadata

Primary sources used for this report:
- `results/eurosat-baseline.ipynb`
- `results/eurosat-efficientnet.ipynb`

Papermill start timestamps in those snapshots:
- Baseline notebook: 2026-04-10T17:51:31
- EfficientNet notebook: 2026-04-10T17:51:51

Approximate total runtime from notebook metadata:
- Baseline notebook: ~97.93 minutes
- EfficientNet notebook: ~82.38 minutes

Execution status:
- Both notebooks completed without fatal runtime exceptions.

## 3. Evaluation Setup Used in These Runs

- Dataset: EuroSAT (10 classes)
- Split strategy: stratified
- Split ratios: train 0.70 / validation 0.15 / test 0.15
- Fixed split seed: 42
- Model selection rule: choose best run by `val_f1_best` on validation split
- Holdout rule: report test metrics after selection, not for selecting among candidate runs

Reported metrics in this report:
- Validation: `val_f1_best`
- Test: accuracy, macro F1
- Additional diagnostics: per-class precision, per-class recall, confusion matrix paths

## 4. Baseline CNN Results (Shared Baseline)

### 4.1 Baseline Ablation Table

| Run ID | Augmentation | Learning Rate | Epochs Ran | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---:|---:|---:|---:|---:|---|
| baseline_reference_none | none | 0.00100 | 50 | 0.915363 | 0.912346 | 0.908949 | no |
| baseline_flips | flips | 0.00100 | 50 | 0.951813 | 0.947901 | 0.946238 | no |
| baseline_flips_low_lr | flips | 0.00050 | 50 | 0.959781 | 0.961975 | 0.960863 | yes |

### 4.2 Baseline Selection Outcome

- Selection pool: all baseline runs
- Selection metric: `val_f1_best`
- Selected run: `baseline_flips_low_lr`

Key output artifacts:
- Summary CSV: `/kaggle/working/artifacts/reports/baseline/baseline_experiment_summary.csv`
- Holdout CSV: `/kaggle/working/artifacts/reports/baseline/baseline_holdout_report.csv`
- Selection manifest: `/kaggle/working/artifacts/reports/baseline/baseline_model_selection.json`
- Selected run report JSON: `/kaggle/working/artifacts/reports/baseline/baseline_flips_low_lr.json`
- Selected checkpoint: `/kaggle/working/checkpoints/baseline_cnn/baseline_flips_low_lr/best_checkpoint.pt`

### 4.3 Baseline Interpretation

- Data augmentation (`flips`) provides a major gain over the no-augmentation reference.
- Lower learning rate improves the already augmented baseline further.
- F1 gain from `baseline_reference_none` to `baseline_flips`: +0.037289.
- Additional F1 gain from `baseline_flips` to `baseline_flips_low_lr`: +0.014625.
- Most difficult class in baseline runs is typically `PermanentCrop` (lowest precision/recall among classes).

## 5. EfficientNetB0 Results (Milos Scope)

### 5.1 EfficientNet Ablation Table

| Run ID | Stage | Augmentation | Learning Rate | Epochs Ran | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---|---:|---:|---:|---:|---:|---|
| efficientnet_stage1_reference | stage1 | flips | 0.00100 | 9 | 0.920765 | 0.926173 | 0.923145 | no |
| efficientnet_stage2_reference | stage2 | flips | 0.00010 | 24 | 0.978875 | 0.974321 | 0.973621 | yes |
| efficientnet_stage2_low_lr | stage2 | flips | 0.00005 | 24 | 0.975499 | 0.971358 | 0.970286 | no |
| efficientnet_stage2_no_aug | stage2 | none | 0.00010 | 13 | 0.965855 | 0.966914 | 0.965938 | no |

### 5.2 EfficientNet Selection Outcome

- Selection pool: Stage 2 runs only
- Selection metric: `val_f1_best`
- Selected run: `efficientnet_stage2_reference`

Key output artifacts:
- Summary CSV: `/kaggle/working/artifacts/reports/efficientnet/efficientnet_experiment_summary.csv`
- Holdout CSV: `/kaggle/working/artifacts/reports/efficientnet/efficientnet_holdout_report.csv`
- Selection manifest: `/kaggle/working/artifacts/reports/efficientnet/efficientnet_model_selection.json`
- Selected run report JSON: `/kaggle/working/artifacts/reports/efficientnet/efficientnet_stage2_reference.json`
- Selected checkpoint: `/kaggle/working/checkpoints/efficientnet_b0/efficientnet_stage2_reference/best_checkpoint.pt`

### 5.3 EfficientNet Interpretation

- Stage 2 fine-tuning materially improves performance over Stage 1 transfer-only setup.
- Macro F1 gain from `stage1_reference` to selected `stage2_reference`: +0.050476.
- In Stage 2 ablation, the reference setting is best.
- `stage2_low_lr` is lower than reference by -0.003335 macro F1.
- `stage2_no_aug` is lower than reference by -0.007683 macro F1.
- Recurring difficult class signal remains around `PermanentCrop`/`River`/`Highway` depending on variant.

## 6. Selected-Model Comparison (Baseline vs EfficientNet)

| Model Family | Selected Run ID | Val Macro F1 (best) | Test Accuracy | Test Macro F1 |
|---|---|---:|---:|---:|
| baseline_cnn | baseline_flips_low_lr | 0.959781 | 0.961975 | 0.960863 |
| efficientnet_b0 | efficientnet_stage2_reference | 0.978875 | 0.974321 | 0.973621 |

Absolute differences (EfficientNet - Baseline):
- Accuracy: +0.012346
- Macro F1: +0.012758

Relative differences (EfficientNet vs Baseline):
- Accuracy: +1.28%
- Macro F1: +1.33%

Validation-to-test gap on selected runs:
- Baseline: +0.001082 macro F1 (test slightly above selected validation best)
- EfficientNet: -0.005254 macro F1 (small expected drop from validation best to holdout test)

## 7. Engineering Quality and Methodology Notes

Strengths:
- Correct anti-leakage model selection is used (`val_f1_best` for selection, test for final reporting).
- Selection decisions are explicitly persisted via selection manifest files.
- Stage-specific selection policy for EfficientNet is explicit and reproducible.

Observed technical caveats in snapshots:
- PyTorch CuBLAS determinism warning appears during training (`CUBLAS_WORKSPACE_CONFIG` not set in Kaggle process).
- `pip install -e .` reports dependency resolver conflicts against preinstalled Kaggle packages.

Practical impact:
- These caveats did not prevent successful execution or change the visible ranking in this run.
- For stronger reproducibility claims in the thesis, environment hardening is recommended (see next section).

## 8. Recommended Next Steps for Final Thesis-Grade Evidence

- Run a small multi-seed replication (for example seeds 42, 43, 44) for selected configurations.
- Report mean and standard deviation for accuracy and macro F1.
- Freeze environment dependencies explicitly for reproducibility appendix.
- Ensure deterministic environment variables are set before training processes in Kaggle session bootstrap.

## 9. ResNet50 Section (Bojan Template)

This section is intentionally left for Bojan, preserving ownership boundaries.

### 9.1 ResNet50 Run Metadata

Fill:
- Notebook/script source:
- Execution timestamp:
- Split seed and split manifest used:
- Stage protocol (if staged):

### 9.2 ResNet50 Ablation Table

| Run ID | Stage | Augmentation | Learning Rate | Epochs Ran | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---|---:|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### 9.3 ResNet50 Selection Record

Fill:
- Selection pool definition:
- Selection metric:
- Selected run ID:
- Selected validation macro F1:
- Holdout report path:
- Selected checkpoint path:

### 9.4 ResNet50 Interpretation Notes

Fill short bullets:
- Main gain driver(s):
- Main failure mode(s):
- Comparison to baseline and EfficientNet:

## 10. Final Integration Checklist (Milos + Bojan)

- Confirm all model families use the same split manifest and seed policy for each comparison batch.
- Keep model selection validation-only for all families.
- Keep test metrics as holdout-only final values.
- Consolidate selected-run comparison table with Baseline, EfficientNet, and ResNet.
- Add multi-seed statistics if produced.
- Reconcile all artifact paths before final thesis writing.
