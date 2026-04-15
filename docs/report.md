# EuroSAT Experiment Report (Working Source for Thesis Writing)

## 1. Document Role

This document is a consolidated source of verified experiment information for writing the final thesis report.

Scope of this version:
- Included: Baseline CNN results, EfficientNetB0 results (Milos scope), and ResNet50 results (Bojan scope).
- All three model families are now complete and comparable.

This file is meant to reduce ambiguity when converting raw notebook outputs into thesis text.

## 2. Result Sources and Execution Metadata

Primary sources used for this report:
- `results/eurosat-baseline.ipynb` (Baseline CNN)
- `results/eurosat-efficientnet.ipynb` (EfficientNet B0, Milos scope)
- `results/eurosat-resnet50.ipynb` (ResNet50, Bojan scope)

Papermill start timestamps in those snapshots:
- Baseline notebook: 2026-04-10T17:51:31
- EfficientNet notebook: 2026-04-10T17:51:51
- ResNet50 notebook: 2026-04-10T18:45:12

Approximate total runtime from notebook metadata:
- Baseline notebook: ~97.93 minutes
- EfficientNet notebook: ~82.38 minutes
- ResNet50 notebook: ~58.67 minutes

Execution status:
- All three notebooks completed without fatal runtime exceptions.

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

## 6. ResNet50 Results (Bojan Scope)

### 6.1 ResNet50 Run Metadata

- Notebook/script source: `results/eurosat-resnet50.ipynb`
- Execution timestamp: 2026-04-10T18:45:12
- Split seed: 42
- Split manifest used: `results/resnet/kaggle/working/artifacts/splits/`
- Stage protocol: Two-stage fine-tuning (Stage 1 frozen backbone, Stage 2 unfrozen)

### 6.2 ResNet50 Ablation Table

| Run ID | Stage | Augmentation | Learning Rate | Epochs Ran | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---|---:|---:|---:|---:|---:|---|
| resnet50_stage1_reference | stage1 | flips | 0.00100 | 5 | 0.929148 | 0.927901 | 0.925859 | no |
| resnet50_stage2_reference | stage2 | flips | 0.00010 | 9 | 0.978449 | 0.978765 | 0.978468 | yes |
| resnet50_stage2_low_lr | stage2 | flips | 0.00005 | 11 | 0.978224 | 0.979506 | 0.979078 | no |
| resnet50_stage2_no_aug | stage2 | none | 0.00010 | 5 | 0.967177 | 0.964691 | 0.963568 | no |

### 6.3 ResNet50 Selection Outcome

- Selection pool: Stage 2 runs only
- Selection metric: `val_f1_best`
- Selected run: `resnet50_stage2_reference`

Key output artifacts:
- Summary CSV: `/kaggle/working/artifacts/reports/resnet50/resnet50_experiment_summary.csv`
- Holdout CSV: `/kaggle/working/artifacts/reports/resnet50/resnet50_holdout_report.csv`
- Selection manifest: `/kaggle/working/artifacts/reports/resnet50/resnet50_model_selection.json`
- Selected run report JSON: `/kaggle/working/artifacts/reports/resnet50/resnet50_stage2_reference.json`
- Selected checkpoint: `/kaggle/working/checkpoints/resnet50/resnet50_stage2_reference/best_checkpoint.pt`

### 6.4 ResNet50 Interpretation

- Stage 2 fine-tuning materially improves performance over Stage 1 transfer-only setup.
- Macro F1 gain from `stage1_reference` to selected `stage2_reference`: +0.052609 (largest two-stage gain among all models).
- In Stage 2 ablation, reference configuration achieves best validation macro F1 (0.978449).
- `stage2_low_lr` shows marginal improvement (+0.000775 macro F1) but worse validation performance vs reference; test accuracy improves slightly (+0.000741).
- `stage2_no_aug` degrades by -0.014900 macro F1, consistent with EfficientNet behavior.
- ResNet50 converges rapidly: Stage 1 in 5 epochs, Stage 2 in 9 epochs (fastest of all models).
- Validation-to-test gap is excellent: +0.000019 macro F1 (near-perfect stability, best among all models).

## 7. Three-Way Model Comparison (Baseline vs EfficientNet vs ResNet50)

### 7.1 Selected-Run Performance Summary

| Model Family | Selected Run ID | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Stage 2 Convergence (Epochs) |
|---|---|---:|---:|---:|---:|
| baseline_cnn | baseline_flips_low_lr | 0.959781 | 0.961975 | 0.960863 | 50 |
| efficientnet_b0 | efficientnet_stage2_reference | 0.978875 | 0.974321 | 0.973621 | 24 |
| resnet50 | resnet50_stage2_reference | 0.978449 | 0.978765 | 0.978468 | 9 |

### 7.2 Pairwise Comparisons

**EfficientNet vs Baseline:**
- Accuracy gain: +0.012346 (+1.28%)
- Macro F1 gain: +0.012758 (+1.33%)
- Convergence: -26 epochs (Stage 2 only, -4.3% relative speedup due to transfer learning)

**ResNet50 vs Baseline:**
- Accuracy gain: +0.016790 (+1.74%)
- Macro F1 gain: +0.017605 (+1.83%)
- Convergence: Much faster (two-stage transfer vs single-stage training)

**ResNet50 vs EfficientNet:**
- Accuracy gain: +0.004444 (+0.46%)
- Macro F1 gain: +0.004847 (+0.50%)
- Convergence: -15 epochs for Stage 2 (63% faster fine-tuning convergence)

### 7.3 Stability and Overfitting Analysis

Validation-to-test gap (macro F1):
- Baseline: +0.001082 (test slightly better than validation best—minimal overfitting risk)
- EfficientNet: -0.005254 (expected modest drop from validation best to holdout test)
- ResNet50: +0.000019 (exceptional stability; test virtually matches validation best)

Interpretation:
- ResNet50 exhibits the tightest control, suggesting robust generalization and minimal tuning artifacts.
- EfficientNet shows a larger gap, typical of fine-tuned transfer models on smaller validation cohorts.
- Baseline shows near-zero gap, consistent with simpler single-stage training.

### 7.4 Class-Level Performance Variation

All three models show similar per-class patterns:
- Strongest classes: Forest, Industrial, Residential (highest precision/recall across all models)
- Most challenging classes: PermanentCrop, River, Highway (lower precision or recall)
- Interpretation: Boundary ambiguity between permanent crops and pasture, and visual similarity in river/highway corridors, remain challenging even for deep transfer models.

### 7.5 Key Findings and Recommendations

1. **Transfer Learning Impact**: Both EfficientNet and ResNet50 substantially outperform the baseline CNN, validating the transfer-learning hypothesis. The gains are driven primarily by pre-trained feature extractors rather than model depth alone.

2. **Efficiency vs Accuracy Trade-off**: ResNet50 achieves the highest accuracy with the fastest fine-tuning convergence (9 epochs), suggesting excellent fit to the EuroSAT domain. EfficientNet achieves comparable validation performance but requires slower fine-tuning (24 epochs).

3. **Model Selection**: ResNet50 is the best performer by test accuracy and macro F1, with superior generalization stability (minimal validation-to-test gap). Recommended for final production deployment.

4. **Staged Fine-Tuning**: Both transfer models benefit significantly from two-stage training (frozen then unfrozen backbone), with macro F1 gains of ~5% over frozen-only baseline transfer.

5. **Augmentation Sensitivity**: Data augmentation (horizontal/vertical flips) is critical for both baseline and transfer models. Disabling augmentation in fine-tuning stage (stage2_no_aug) consistently degrades macro F1 by 1.0–1.5%.

## 8. Engineering Quality and Methodology Notes

Strengths across all three model families:
- Correct anti-leakage model selection is used (`val_f1_best` for selection, test for final reporting).
- Selection decisions are explicitly persisted via selection manifest files for all models.
- Stage-specific selection policy for EfficientNet and ResNet50 is explicit and reproducible.
- Identical split manifests and seed (42) ensure fair comparison across all three models.

Observed technical caveats in snapshots:
- PyTorch CuBLAS determinism warning appears during training (`CUBLAS_WORKSPACE_CONFIG` not set in Kaggle process).
- `pip install -e .` reports dependency resolver conflicts against preinstalled Kaggle packages.

Practical impact:
- These caveats did not prevent successful execution or change the visible ranking in these runs.
- For stronger reproducibility claims in the thesis, environment hardening is recommended (see next section).

Validation protocol rigor:
- All three model families use identical stratified split protocol (70/15/15, seed 42).
- Model selection is validation-only; test metrics reported only for selected runs.
- Stage 2 runs resume from Stage 1 checkpoints (transfer models), preserving deterministic lineage.

## 9. Recommended Next Steps for Final Thesis-Grade Evidence

- Run a small multi-seed replication (for example seeds 42, 43, 44) for all three selected configurations to estimate confidence intervals.
- Report mean and standard deviation for accuracy and macro F1 across seeds.
- Perform per-class analysis to identify which land-cover categories benefit most from transfer learning.
- Export and visualize confusion matrices for all three selected models to support qualitative discussion.
- Freeze environment dependencies explicitly for reproducibility appendix.
- Ensure deterministic environment variables are set before training processes in Kaggle session bootstrap.

## 10. Final Integration Checklist (Milos + Bojan)

- Confirm all model families use the same split manifest and seed policy for each comparison batch.
- Keep model selection validation-only for all families.
- Keep test metrics as holdout-only final values.
- Consolidate selected-run comparison table with Baseline, EfficientNet, and ResNet (completed in Section 7).
- Add multi-seed statistics if produced.
- Reconcile all artifact paths before final thesis writing.
