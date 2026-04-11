# ResNet50 Integration Contract

This document defines the ResNet50 implementation boundary and runtime contract for the EuroSAT land-use classification project in Python.

## Scope

- Owner: Bojan Živanić.
- Model family: ResNet50 fine-tuning on EuroSAT RGB.
- Shared infrastructure remains unchanged: domain/application/infrastructure/entrypoints layering is preserved.

## Integration Points

- Model adapter file: `src/eurosat_classifier/infrastructure/models/resnet50.py`.
- Model registration mechanism: `src/eurosat_classifier/infrastructure/models/registry.py`.
- Shared model factory: `src/eurosat_classifier/infrastructure/models/factory.py`.
- Shared training orchestration: `src/eurosat_classifier/application/services/training_orchestrator.py`.
- Shared dataloaders and transforms: `src/eurosat_classifier/infrastructure/training/split_json_loader.py`.

## Required Model Contract

The ResNet50 adapter must satisfy the same runtime expectations as existing models:

- Expose `num_classes` as an integer attribute.
- Implement `forward(inputs: torch.Tensor) -> torch.Tensor`.
- Support staged fine-tuning through `set_backbone_trainable(trainable: bool)`.
- Keep BatchNorm layers in evaluation mode when backbone is frozen.

## Supported Configuration Keys

Model name:

- `model.name = "resnet50"`

Model options:

- `model.options.use_pretrained` (`true`/`false`)
- `model.options.freeze_backbone` (`true`/`false`)

Training options (shared contract):

- `training.epochs`
- `training.batch_size`
- `training.early_stopping_patience`
- `training.early_stopping_min_delta`
- `training.learning_rate`
- `training.scheduler_factor`
- `training.scheduler_patience`
- `training.min_learning_rate`
- `training.augmentation_mode`
- `training.resume_from` (stage 2)

## Staged Fine-Tuning Flow

1. Stage 1:
   - Use pretrained ResNet50.
   - Freeze backbone and train classifier head.
   - Save best checkpoint.
2. Stage 2:
   - Load stage 1 checkpoint via `training.resume_from`.
   - Unfreeze backbone.
   - Continue fine-tuning with lower learning rate.

## Runtime Assets

- Stage 1 config: `configs/resnet50.stage1.json`.
- Stage 2 config: `configs/resnet50.stage2.json`.
- Reusable template: `configs/resnet50.template.json`.

## Checkpoint Compatibility Rule

- `training.resume_from` must point to a checkpoint created from the same model architecture (`resnet50`).
- Loading checkpoints from other architectures is rejected by the shared checkpoint store.

## Example CLI Commands

The following are example commands for local runs:

```powershell
$env:PYTHONPATH='src'; python run.py --run-baseline --config configs/resnet50.stage1.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits --reports-output artifacts/reports/resnet50_stage1_final.json --checkpoints-output checkpoints/resnet50/stage1
$env:PYTHONPATH='src'; python run.py --run-baseline --config configs/resnet50.stage2.json --defaults configs/experiment.defaults.json --splits-output artifacts/splits --reports-output artifacts/reports/resnet50_stage2_final.json --checkpoints-output checkpoints/resnet50/stage2
```
