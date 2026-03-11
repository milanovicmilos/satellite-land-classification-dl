# EuroSAT Implementation TODO (Production-Oriented)

This file is the execution roadmap for the EuroSAT land-use classification project in Python.

Project context:
- Dataset: EuroSAT RGB JPEG (`data/EuroSAT`)
- Split: stratified 70/15/15 with fixed seed
- Metrics: accuracy, macro F1-score, confusion matrix, per-class precision, per-class recall
- Team ownership: Milos (EfficientNetB0 + shared infrastructure), Bojan (ResNet50)

## 1. Lock Scope And Reproducibility Rules

Goal:
- Freeze experiment conventions so all models are compared fairly and reproducibly.

Work items:
- [x] Finalize canonical split seed and persist split metadata artifact.
- [x] Define single source for model comparison table schema.
- [x] Document acceptance criteria for baseline and EfficientNetB0 milestones.

Folders and files:
- Existing: `specifikacija.txt`
- Existing: `README.md`
- Existing: `configs/baseline.example.json`
- Planned: `configs/experiment.defaults.json`
- Planned: `artifacts/splits/split_manifest.json`
- Planned: `docs/evaluation_protocol.md`

Implementation status:
- Completed in this phase:
	- `configs/experiment.defaults.json`
	- `artifacts/splits/split_manifest.json`
	- `docs/evaluation_protocol.md`
	- `src/eurosat_classifier/infrastructure/config_loader.py` (defaults merge)
	- `src/eurosat_classifier/entrypoints/cli.py` (`--defaults` support)
	- `tests/test_config_loader.py` and `tests/test_cli.py` updates for validation coverage

## 2. Data Validation And Deterministic Split Pipeline

Goal:
- Ensure dataset integrity before training and produce deterministic train/val/test splits.

Work items:
- [x] Implement dataset index builder from `data/EuroSAT` folder structure.
- [x] Add input validation (class presence, file count, extension checks, corrupt image detection).
- [x] Implement stratified split generator with fixed seed.
- [x] Save split files for reuse across all models.

Folders and files:
- Existing: `data/EuroSAT`
- Existing: `data/README.md`
- Existing: `src/eurosat_classifier/domain/entities.py`
- Existing: `src/eurosat_classifier/application/contracts.py`
- Existing: `src/eurosat_classifier/application/use_cases.py`
- Planned: `src/eurosat_classifier/infrastructure/datasets/eurosat_index.py`
- Planned: `src/eurosat_classifier/infrastructure/datasets/splitter.py`
- Planned: `tests/test_dataset_index.py`
- Planned: `tests/test_splitter.py`

Implementation status:
- Completed in this phase:
	- `src/eurosat_classifier/infrastructure/datasets/eurosat_index.py`
	- `src/eurosat_classifier/infrastructure/datasets/splitter.py`
	- `src/eurosat_classifier/infrastructure/datasets/split_store.py`
	- `src/eurosat_classifier/application/contracts.py` (dataset prep contracts)
	- `src/eurosat_classifier/application/use_cases.py` (`PrepareDataset` use case)
	- `src/eurosat_classifier/entrypoints/cli.py` (`--prepare-dataset`, `--splits-output`)
	- `tests/test_dataset_index.py`
	- `tests/test_splitter.py`
	- `tests/test_cli.py` updates for new CLI flags

Produced artifacts:
- `artifacts/splits/train_split.json`
- `artifacts/splits/validation_split.json`
- `artifacts/splits/test_split.json`
- `artifacts/splits/split_summary.json`

## 3. Baseline CNN End-to-End (Reference System)

Goal:
- Deliver the complete baseline training and evaluation flow used as reference for all comparisons.

Work items:
- [x] Implement baseline CNN model module.
- [x] Implement training loop with early stopping and checkpoint saving.
- [x] Implement evaluation on test split with required metrics.
- [x] Export baseline result summary artifact.

Folders and files:
- Existing: `src/eurosat_classifier/infrastructure/models/registry.py`
- Existing: `src/eurosat_classifier/application/use_cases.py`
- Existing: `src/eurosat_classifier/entrypoints/cli.py`
- Planned: `src/eurosat_classifier/infrastructure/models/baseline_cnn.py`
- Planned: `src/eurosat_classifier/application/use_cases/train_baseline.py`
- Planned: `src/eurosat_classifier/application/use_cases/evaluate_model.py`
- Planned: `src/eurosat_classifier/infrastructure/training/trainer.py`
- Planned: `src/eurosat_classifier/infrastructure/checkpointing/store.py`
- Planned: `artifacts/reports/baseline_metrics.json`

Implementation status:
- `src/eurosat_classifier/infrastructure/models/baseline_cnn.py` (real CNN model)
- `src/eurosat_classifier/infrastructure/models/factory.py`
- `src/eurosat_classifier/infrastructure/training/split_json_loader.py` (PyTorch dataloaders)
- `src/eurosat_classifier/infrastructure/training/baseline_trainer.py` (real train loop)
- `src/eurosat_classifier/infrastructure/evaluation/baseline_evaluator.py` (forward-pass evaluation)
- `src/eurosat_classifier/infrastructure/checkpointing/store.py` (best `.pt` checkpoint + metadata)
- `src/eurosat_classifier/entrypoints/cli.py` (`--run-baseline` flow)
- `artifacts/reports/baseline_metrics.json` and `artifacts/reports/baseline_smoke_metrics.json` generated via CLI

## 4. Shared Training Engine And Evaluation Core

Goal:
- Avoid duplicated logic and keep architecture clean across baseline, EfficientNetB0, and ResNet50.

Work items:
- [x] Define stable interfaces for model factory, dataloaders, trainer, evaluator, checkpoint store.
- [x] Centralize metric calculation and report serialization.
- [x] Keep framework-specific details in infrastructure layer only.

Folders and files:
- Existing: `src/eurosat_classifier/application/contracts.py`
- Existing: `src/eurosat_classifier/domain/metrics.py`
- Existing: `src/eurosat_classifier/infrastructure/logging.py`
- Planned: `src/eurosat_classifier/domain/metrics_calculator.py`
- Planned: `src/eurosat_classifier/application/services/training_orchestrator.py`
- Planned: `src/eurosat_classifier/infrastructure/evaluation/report_writer.py`
- Planned: `tests/test_metrics_calculator.py`

Implementation status:
- `src/eurosat_classifier/application/contracts.py` (shared engine protocols)
- `src/eurosat_classifier/domain/metrics_calculator.py`
- `src/eurosat_classifier/application/services/training_orchestrator.py`
- `src/eurosat_classifier/infrastructure/evaluation/report_writer.py`
- `src/eurosat_classifier/infrastructure/reproducibility.py` (`set_seed` for random, NumPy, Torch CPU/CUDA, deterministic cuDNN)
- `src/eurosat_classifier/application/services/training_orchestrator.py` (global seed applied from config split seed at the start of each run)
- `src/eurosat_classifier/application/services/training_orchestrator.py` (CLI payload now includes confusion matrix for report parity)
- `tests/test_metrics_calculator.py`
- `tests/test_report_writer.py`
- `tests/test_training_orchestrator.py`
- Smoke reproducibility verified with two consecutive baseline smoke runs producing identical loss/accuracy/F1 outputs

## 5. EfficientNetB0 Fine-Tuning (Milos Scope)

Goal:
- Implement and tune EfficientNetB0 using the shared engine and compare against baseline.

Work items:
- [ ] Add EfficientNetB0 model adapter/factory implementation.
- [ ] Add staged fine-tuning strategy (frozen backbone -> progressive unfreeze).
- [ ] Add model-specific config presets.
- [ ] Generate comparison report versus baseline.

Folders and files:
- Existing: `src/eurosat_classifier/infrastructure/models/registry.py`
- Existing: `configs/baseline.example.json`
- Planned: `src/eurosat_classifier/infrastructure/models/efficientnet_b0.py`
- Planned: `configs/efficientnet_b0.stage1.json`
- Planned: `configs/efficientnet_b0.stage2.json`
- Planned: `artifacts/reports/efficientnet_b0_metrics.json`
- Planned: `artifacts/reports/baseline_vs_efficientnet_b0.md`

## 6. ResNet50 Integration Boundary (Bojan Scope)

Goal:
- Keep clean extension points ready for Bojan's implementation without crossing ownership boundaries.

Work items:
- [ ] Keep interface and registry hooks ready for ResNet50.
- [ ] Provide config template and integration checklist.
- [ ] Do not implement Bojan-owned model internals unless explicitly requested.

Folders and files:
- Existing: `src/eurosat_classifier/infrastructure/models/registry.py`
- Existing: `.github/copilot-instructions.md`
- Planned: `configs/resnet50.template.json`
- Planned: `docs/resnet50_integration_contract.md`

## 7. Quality Gates (Tests, Lint, Type Checks)

Goal:
- Enforce reliability and detect regressions before long training runs.

Work items:
- [ ] Expand unit tests for split logic, config validation, and metrics.
- [ ] Add smoke tests for CLI and short training path.
- [ ] Add lint and type-check tooling configuration.

Folders and files:
- Existing: `tests/test_cli.py`
- Existing: `tests/test_config_loader.py`
- Existing: `pyproject.toml`
- Planned: `tests/test_training_smoke.py`
- Planned: `pyproject.toml` (tool sections for lint/type/test)
- Planned: `requirements-dev.txt`
- Planned: `.github/workflows/ci.yml`

## 8. Packaging, Inference, And Production Rollout

Goal:
- Prepare a stable artifact and prediction interface suitable for deployment.

Work items:
- [ ] Define model artifact versioning and metadata schema.
- [ ] Implement inference entrypoint (batch or API).
- [ ] Add rollback-safe model selection mechanism.
- [ ] Document operational playbook.

Folders and files:
- Existing: `run.py`
- Existing: `src/eurosat_classifier/entrypoints/cli.py`
- Planned: `src/eurosat_classifier/entrypoints/predict.py`
- Planned: `src/eurosat_classifier/application/use_cases/run_inference.py`
- Planned: `artifacts/models/` (local runtime artifacts)
- Planned: `docs/production_runbook.md`

## 9. Copilot Context And Instruction Continuity

Goal:
- Keep future chat sessions aligned with architecture, language, and ownership constraints.

Work items:
- [ ] Keep project-wide instructions current with architecture and workflow changes.
- [ ] Keep Python and documentation instruction files aligned with actual implementation.
- [ ] Update this TODO after every major milestone.

Folders and files:
- Existing: `.github/copilot-instructions.md`
- Existing: `.github/instructions/python-architecture.instructions.md`
- Existing: `.github/instructions/project-docs.instructions.md`
- Existing: `TODO.md`

## Current Phase Recommendation

Start with Phase 5.

Reason:
- Phases 1, 2, 3, and 4 are completed and validated.
- Reproducibility and CLI/report consistency are now enforced in the shared engine.
- Phase 5 delivers your owned advanced model (EfficientNetB0) on top of stable shared infrastructure.