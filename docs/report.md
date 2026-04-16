# EuroSAT Project Master Report (Thesis Source Document)

## 1. Document Role and Usage Rules

This file is not the final scientific paper. It is a complete source document for writing the IEEE-format paper without reopening the codebase.

Primary intent:
- Keep all project facts, protocol details, implementation notes, and verified metrics in one place.
- Separate verified facts from proposed additions.
- Preserve enough context to write Introduction, Related Work, Methodology, Results, and Discussion directly from this file.

Scope of this version:
- Included model families: baseline CNN, EfficientNetB0 (Milos scope), and ResNet50 (Bojan scope).
- Included evidence type: execution metadata, selection protocol, ablations, cross-model comparison, engineering caveats, writing blueprint.

Authoring rule for thesis drafting:
- Treat every numeric claim in the final paper as grounded in this report.
- If a claim is not in this report, add it here first.

## 2. Project Summary for Introduction Section

### 2.1 Problem Definition

The project addresses 10-class land-use classification from satellite images using the EuroSAT dataset.

Input-output formulation:
- Input: RGB satellite image patch (64x64).
- Output: one label from 10 land-use classes.

### 2.2 Why the Problem Matters

Land-use classification is important for:
- urban planning and infrastructure monitoring,
- agricultural analysis,
- forest and water-area monitoring,
- environmental and climate-related decision support.

Automated classification improves speed and consistency compared to manual analysis and enables scalable monitoring pipelines.

### 2.3 Project Goal

Evaluate whether transfer learning with modern pretrained CNN backbones improves performance over a custom baseline CNN under a strict and fair protocol.

Research comparison axis:
- Baseline CNN (from scratch) versus EfficientNetB0 and ResNet50 (two-stage transfer learning).

### 2.4 Team Responsibilities

- Milos Milanovic: EfficientNetB0 and shared infrastructure.
- Bojan Zivanic: ResNet50.
- Shared responsibilities: baseline, protocol design, reporting conventions, reproducibility setup.

## 3. Research Questions and Hypotheses

### 3.1 Research Questions

RQ1: Do transfer-learning models outperform the baseline CNN on EuroSAT under identical split and selection rules?

RQ2: How much does staged fine-tuning (frozen then unfrozen) contribute over frozen-only transfer setup?

RQ3: How sensitive are model families to augmentation and learning-rate variants?

RQ4: Which model gives the best trade-off between final test performance and convergence speed?

### 3.2 Working Hypotheses

H1: EfficientNetB0 and ResNet50 will outperform baseline CNN on holdout test accuracy and macro F1.

H2: Stage 2 fine-tuning will outperform Stage 1 frozen setup for both transfer models.

H3: Removing augmentation will degrade macro F1 for all model families.

## 4. Dataset and Label Space

Dataset:
- EuroSAT RGB version (10 classes, ~27,000 images).

Classes:
- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

Dataset assumptions used in this project:
- Labels are inferred from folder structure and exported split manifests.
- No additional manual annotation was required.

Primary local data locations:
- `data/EuroSAT/`
- `data/EuroSAT/train.csv`
- `data/EuroSAT/validation.csv`
- `data/EuroSAT/test.csv`

## 5. Experimental Protocol (Canonical)

### 5.1 Split and Reproducibility Policy

- Strategy: stratified split by class.
- Ratios: 70% train / 15% validation / 15% test.
- Fixed seed: 42.
- Split artifacts are persisted and reused across model families.

Required split artifacts:
- `artifacts/splits/split_manifest.json`
- `artifacts/splits/train_split.json`
- `artifacts/splits/validation_split.json`
- `artifacts/splits/test_split.json`
- `artifacts/splits/split_summary.json`

### 5.2 Selection and Holdout Rules

Anti-leakage rule:
- Model selection is done by validation metric only (`val_f1_best`).
- Test split is used only once for final holdout reporting of selected runs.

Selection pools:
- Baseline CNN: all baseline variants.
- EfficientNetB0: Stage 2 variants only.
- ResNet50: Stage 2 variants only.

### 5.3 Required Reported Metrics

- Test accuracy.
- Test macro F1.
- Per-class precision.
- Per-class recall.
- Confusion matrix.

## 6. Methodology and Implementation Summary

### 6.1 Architecture and Code Organization

The repository follows Clean Architecture-style separation:
- Domain layer: entities and metric contracts.
- Application layer: use cases and orchestration.
- Infrastructure layer: config loading, datasets, model wiring, logging, checkpointing.
- Entrypoints: CLI execution flow.

Key locations:
- `src/eurosat_classifier/domain/`
- `src/eurosat_classifier/application/`
- `src/eurosat_classifier/infrastructure/`
- `src/eurosat_classifier/entrypoints/`

### 6.2 Model Families and Training Strategy

Baseline CNN:
- Single-stage training from scratch.
- Serves as reference comparator.

EfficientNetB0 and ResNet50:
- Stage 1: frozen backbone transfer setup.
- Stage 2: unfrozen fine-tuning resumed from Stage 1 checkpoint.
- Same split policy and reporting schema as baseline.

### 6.3 Configuration and Execution Style

Config-first workflow:
- JSON configurations define model, training, and augmentation settings.
- CLI supports config overrides for controlled sweeps.

Active config files:
- `configs/baseline_cnn.json`
- `configs/efficientnet_b0.stage1.json`
- `configs/efficientnet_b0.stage2.json`
- `configs/resnet50.stage1.json`
- `configs/resnet50.stage2.json`
- `configs/experiment.defaults.json`

### 6.4 Operational Pipeline (High-Level)

1. Prepare or reuse deterministic split artifacts.
2. Train candidate runs per model family.
3. Select winner by validation macro F1.
4. Export holdout metrics and artifacts for selected run.
5. Aggregate three-way comparison table.

### 6.5 Engineering Constraints and Caveats

Observed in Kaggle snapshots:
- `pip install -e .` may warn about resolver conflicts with preinstalled packages.
- PyTorch may emit CuBLAS determinism warning if `CUBLAS_WORKSPACE_CONFIG` is not set.

Impact in current runs:
- No fatal runtime exceptions.
- Ranking conclusions remained stable in these snapshots.

## 7. Related Work Notes (Paper Input)

This section is designed as a writing-ready source for the IEEE related-work chapter.

### 7.1 Minimum Literature Categories to Cover

Include at least one representative work per category:
- EuroSAT dataset benchmark paper.
- Remote-sensing land-use classification using CNNs.
- Transfer-learning studies in remote sensing.
- Backbone-focused papers (ResNet and EfficientNet original architecture papers).

### 7.2 Suggested Anchor References

- Helber et al., EuroSAT dataset benchmark (core dataset citation).
- He et al., ResNet architecture.
- Tan and Le, EfficientNet architecture.

### 7.3 How to Position This Project Against Prior Work

Recommended comparison narrative:
- This project focuses on controlled protocol symmetry (same split manifest, same selection rule).
- It compares baseline and transfer approaches under the same evaluation contract.
- It emphasizes reproducibility and anti-leakage model selection as a methodological contribution.

### 7.4 Related Work Writing Checklist

Before final paper submission, ensure each cited paper has:
- complete bibliographic metadata,
- clear statement of task and dataset,
- explicit relevance to this project,
- concise contrast to your setup.

## 8. Result Sources and Execution Metadata

Primary result snapshots used in this report:
- `results/eurosat-baseline.ipynb` (Baseline CNN)
- `results/eurosat-efficientnet.ipynb` (EfficientNetB0)
- `results/eurosat-resnet.ipynb` (ResNet50 runs and exports)

Papermill start timestamps:
- Baseline notebook: 2026-04-10T17:51:31
- EfficientNet notebook: 2026-04-10T17:51:51
- ResNet notebook: 2026-04-10T18:45:12

Approximate runtime from notebook metadata:
- Baseline: ~97.93 minutes
- EfficientNet: ~82.38 minutes
- ResNet50: ~58.67 minutes

Execution status:
- All three notebook pipelines completed without fatal runtime exceptions.

## 9. Baseline CNN Results (Shared Baseline)

### 9.1 Baseline Ablation Table

| Run ID | Augmentation | Learning Rate | Epochs Ran | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---:|---:|---:|---:|---:|---|
| baseline_reference_none | none | 0.00100 | 50 | 0.915363 | 0.912346 | 0.908949 | no |
| baseline_flips | flips | 0.00100 | 50 | 0.951813 | 0.947901 | 0.946238 | no |
| baseline_flips_low_lr | flips | 0.00050 | 50 | 0.959781 | 0.961975 | 0.960863 | yes |

### 9.2 Baseline Selection Outcome

- Selection pool: all baseline runs.
- Selection metric: `val_f1_best`.
- Selected run: `baseline_flips_low_lr`.

Key artifacts:
- `/kaggle/working/artifacts/reports/baseline/baseline_experiment_summary.csv`
- `/kaggle/working/artifacts/reports/baseline/baseline_holdout_report.csv`
- `/kaggle/working/artifacts/reports/baseline/baseline_model_selection.json`
- `/kaggle/working/artifacts/reports/baseline/baseline_flips_low_lr.json`
- `/kaggle/working/checkpoints/baseline_cnn/baseline_flips_low_lr/best_checkpoint.pt`

### 9.3 Baseline Interpretation

- Augmentation (`flips`) gives a major gain over no-augmentation reference.
- Lower learning rate gives an additional gain on top of augmentation.
- Macro F1 gain (`reference_none` -> `flips`): +0.037289.
- Macro F1 gain (`flips` -> `flips_low_lr`): +0.014625.

## 10. EfficientNetB0 Results (Milos Scope)

### 10.1 EfficientNetB0 Ablation Table

| Run ID | Stage | Augmentation | Learning Rate | Epochs Ran | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---|---:|---:|---:|---:|---:|---|
| efficientnet_stage1_reference | stage1 | flips | 0.00100 | 9 | 0.920765 | 0.926173 | 0.923145 | no |
| efficientnet_stage2_reference | stage2 | flips | 0.00010 | 24 | 0.978875 | 0.974321 | 0.973621 | yes |
| efficientnet_stage2_low_lr | stage2 | flips | 0.00005 | 24 | 0.975499 | 0.971358 | 0.970286 | no |
| efficientnet_stage2_no_aug | stage2 | none | 0.00010 | 13 | 0.965855 | 0.966914 | 0.965938 | no |

### 10.2 EfficientNetB0 Selection Outcome

- Selection pool: Stage 2 runs only.
- Selection metric: `val_f1_best`.
- Selected run: `efficientnet_stage2_reference`.

Key artifacts:
- `/kaggle/working/artifacts/reports/efficientnet/efficientnet_experiment_summary.csv`
- `/kaggle/working/artifacts/reports/efficientnet/efficientnet_holdout_report.csv`
- `/kaggle/working/artifacts/reports/efficientnet/efficientnet_model_selection.json`
- `/kaggle/working/artifacts/reports/efficientnet/efficientnet_stage2_reference.json`
- `/kaggle/working/checkpoints/efficientnet_b0/efficientnet_stage2_reference/best_checkpoint.pt`

### 10.3 EfficientNetB0 Interpretation

- Stage 2 fine-tuning clearly improves over Stage 1.
- Macro F1 gain (`stage1_reference` -> `stage2_reference`): +0.050476.
- In Stage 2 ablation, reference setting remains best by validation macro F1.
- `stage2_low_lr` is lower than reference by -0.003335 macro F1.
- `stage2_no_aug` is lower than reference by -0.007683 macro F1.

## 11. ResNet50 Results (Bojan Scope)

### 11.1 ResNet50 Run Metadata

- Snapshot source: `results/eurosat-resnet.ipynb`.
- Execution timestamp: 2026-04-10T18:45:12.
- Split seed: 42.
- Split manifest location in run outputs: `results/resnet/kaggle/working/artifacts/splits/`.
- Stage protocol: two-stage fine-tuning (frozen -> unfrozen).

### 11.2 ResNet50 Ablation Table

| Run ID | Stage | Augmentation | Learning Rate | Epochs Ran | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Selected For Final |
|---|---|---|---:|---:|---:|---:|---:|---|
| resnet50_stage1_reference | stage1 | flips | 0.00100 | 5 | 0.929148 | 0.927901 | 0.925859 | no |
| resnet50_stage2_reference | stage2 | flips | 0.00010 | 9 | 0.978449 | 0.978765 | 0.978468 | yes |
| resnet50_stage2_low_lr | stage2 | flips | 0.00005 | 11 | 0.978224 | 0.979506 | 0.979078 | no |
| resnet50_stage2_no_aug | stage2 | none | 0.00010 | 5 | 0.967177 | 0.964691 | 0.963568 | no |

### 11.3 ResNet50 Selection Outcome

- Selection pool: Stage 2 runs only.
- Selection metric: `val_f1_best`.
- Selected run: `resnet50_stage2_reference`.

Key artifacts:
- `/kaggle/working/artifacts/reports/resnet50/resnet50_experiment_summary.csv`
- `/kaggle/working/artifacts/reports/resnet50/resnet50_holdout_report.csv`
- `/kaggle/working/artifacts/reports/resnet50/resnet50_model_selection.json`
- `/kaggle/working/artifacts/reports/resnet50/resnet50_stage2_reference.json`
- `/kaggle/working/checkpoints/resnet50/resnet50_stage2_reference/best_checkpoint.pt`

### 11.4 ResNet50 Interpretation

- Stage 2 fine-tuning gives strong improvement over Stage 1.
- Macro F1 gain (`stage1_reference` -> `stage2_reference`): +0.052609.
- `stage2_low_lr` has slightly higher test macro F1 (+0.000610) but lower validation macro F1 (-0.000225) than reference.
- `stage2_no_aug` drops by -0.014900 macro F1 relative to `stage2_reference`.
- Convergence speed is highest among compared model families.

## 12. Three-Way Model Comparison

### 12.1 Selected-Run Performance Summary

| Model Family | Selected Run ID | Val Macro F1 (best) | Test Accuracy | Test Macro F1 | Stage 2 / Total Epochs |
|---|---|---:|---:|---:|---:|
| baseline_cnn | baseline_flips_low_lr | 0.959781 | 0.961975 | 0.960863 | 50 |
| efficientnet_b0 | efficientnet_stage2_reference | 0.978875 | 0.974321 | 0.973621 | 24 (33 incl. Stage 1) |
| resnet50 | resnet50_stage2_reference | 0.978449 | 0.978765 | 0.978468 | 9 (14 incl. Stage 1) |

### 12.2 Pairwise Deltas (Selected Runs)

EfficientNetB0 vs Baseline:
- Accuracy: +0.012346.
- Macro F1: +0.012758.

ResNet50 vs Baseline:
- Accuracy: +0.016790.
- Macro F1: +0.017605.

ResNet50 vs EfficientNetB0:
- Accuracy: +0.004444.
- Macro F1: +0.004847.

### 12.3 Stability Analysis (Validation vs Test)

Validation-to-test macro F1 gap for selected runs:
- Baseline: +0.001082.
- EfficientNetB0: -0.005254.
- ResNet50: +0.000019.

Interpretation:
- ResNet50 shows strongest stability between validation best and test holdout.
- EfficientNetB0 still performs strongly but has a larger expected generalization drop.

### 12.4 Class-Level Difficulty Pattern

Consistent pattern across model families:
- Generally stronger classes: Forest, Industrial, Residential.
- Recurrently difficult classes: PermanentCrop, River, Highway.

Working explanation for discussion section:
- Spectral/texture similarity and boundary ambiguity between some agricultural and corridor-like classes likely drive confusion.

## 13. Error Analysis and Discussion Inputs

Use these points when writing the Discussion section:

1. Transfer-learning gains are substantial and consistent across both advanced backbones.
2. Two-stage fine-tuning is necessary; frozen-only transfer is clearly weaker.
3. Augmentation remains critical even for transfer models.
4. ResNet50 achieved best holdout metrics and fastest convergence in this experiment set.
5. Validation-only selection prevented test-leakage during model choice.

Discussion boundaries:
- Avoid claiming universal superiority beyond EuroSAT and current seed unless multi-seed evidence is added.
- Keep claims tied to measured quantities in this report.

## 14. Threats to Validity and Reproducibility

Internal validity risks:
- Single-seed primary comparison (seed 42) may underrepresent run-to-run variance.
- Notebook environment differences can influence exact determinism.

External validity risks:
- Results are dataset-specific (EuroSAT RGB, 64x64).
- Transferability to other remote-sensing datasets is not yet measured.

Mitigations already in place:
- Fixed stratified split and persisted manifests.
- Uniform selection policy and holdout reporting across model families.
- Explicit per-run artifacts and checkpoints.

## 15. IEEE Paper Blueprint (Professor Requirements Mapped)

Professor-required structure and what to take from this report:

### 15.1 Introduction and Motivation

Source sections in this report:
- Section 2 (problem, importance, goal).
- Section 3 (research questions and hypotheses).

### 15.2 Related Work

Source sections in this report:
- Section 7 (categories, anchor references, positioning).

### 15.3 Methodology and Implementation

Source sections in this report:
- Section 4 (dataset).
- Section 5 (protocol).
- Section 6 (architecture and pipeline).

### 15.4 Results and Discussion

Source sections in this report:
- Sections 8-13 (metadata, ablations, comparisons, interpretation, error analysis).

### 15.5 Conclusion and Future Work

Source sections in this report:
- Section 13 (main conclusions).
- Section 16 (future work checklist).

## 16. Stanford Writing Guidance (Actionable Notes)

Based on Jennifer Widom's "Tips for Writing Technical Papers":

1. Introduction should answer five questions explicitly:
- What is the problem?
- Why is it important?
- Why is it hard?
- Why existing approaches are insufficient?
- What are your key contributions and limitations?

2. Add a concise "Summary of Contributions" list at the end of Introduction.

3. Keep a linear story in the body:
- preliminaries -> method -> experiments -> interpretation.

4. Experiments should show:
- absolute performance,
- relative performance to baseline,
- sensitivity to major parameters.

5. Conclusions must summarize findings without repeating abstract text verbatim.

6. Future work should be concrete and listed as bullets.

7. Writing quality rules:
- define terminology before use,
- avoid vague statements,
- ensure citation consistency and completeness.

## 17. Future Work and Finalization Checklist

Recommended final evidence upgrades before thesis lock:
- Run multi-seed evaluation (for example 42, 43, 44) for selected runs.
- Report mean and standard deviation for accuracy and macro F1.
- Add confusion-matrix figures for selected runs in paper-ready format.
- Freeze environment details and dependency versions for appendix.
- Expand related-work references with full bibliography metadata.

Final pre-submission checklist:
- Confirm all tables use the same protocol and seed policy.
- Confirm all selected runs are validation-selected, test-reported.
- Ensure consistent naming of result snapshots and artifact paths.
- Ensure every numeric claim in the paper exists in this report.

## 18. Appendix: Quick Artifact Index

Core report-oriented files:
- `docs/evaluation_protocol.md`
- `docs/experiments_log.md`
- `results/eurosat-baseline.ipynb`
- `results/eurosat-efficientnet.ipynb`
- `results/eurosat-resnet.ipynb`

Split and reproducibility assets:
- `artifacts/splits/split_manifest.json`
- `artifacts/splits/train_split.json`
- `artifacts/splits/validation_split.json`
- `artifacts/splits/test_split.json`

Notebook templates used for execution flows:
- `notebooks/eurosat_baseline_kaggle.ipynb`
- `notebooks/eurosat_efficientnet_kaggle.ipynb`
- `notebooks/eurosat_resnet50_kaggle.ipynb`
