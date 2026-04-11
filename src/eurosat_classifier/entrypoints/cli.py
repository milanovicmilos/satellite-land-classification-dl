"""CLI entrypoint for scaffold verification and future orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eurosat_classifier.application.use_cases import PrepareDataset, StartTraining
from eurosat_classifier.application.services.training_orchestrator import TrainingOrchestrator
from eurosat_classifier.infrastructure.config_loader import JsonConfigLoader
from eurosat_classifier.infrastructure.checkpointing.store import JsonCheckpointStore
from eurosat_classifier.infrastructure.datasets.eurosat_index import (
    EXPECTED_EUROSAT_CLASSES,
    EuroSatDatasetIndexer,
)
from eurosat_classifier.infrastructure.datasets.split_store import JsonSplitPersistence
from eurosat_classifier.infrastructure.datasets.splitter import StratifiedSplitter
from eurosat_classifier.infrastructure.evaluation.baseline_evaluator import BaselineEvaluator
from eurosat_classifier.infrastructure.evaluation.report_writer import JsonReportWriter
from eurosat_classifier.infrastructure.logging import configure_logging
from eurosat_classifier.infrastructure.models.factory import SharedModelFactory
from eurosat_classifier.infrastructure.training.baseline_trainer import BaselineTrainer
from eurosat_classifier.infrastructure.training.split_json_loader import SplitJsonLoaderFactory


class DryRunTrainingRunner:
    """Non-framework runner used to validate the project scaffold."""

    def run(self, experiment) -> str:
        payload = {
            "experiment_name": experiment.name,
            "dataset_root": experiment.dataset_root,
            "model_name": experiment.model_name,
            "seed": experiment.split.seed,
            "stratified": experiment.split.stratified,
        }
        return json.dumps(payload, indent=2)


def _build_config_overrides(args: argparse.Namespace) -> dict[str, Any] | None:
    """Builds selective config overrides from CLI arguments."""

    overrides: dict[str, Any] = {}

    if args.experiment_name is not None:
        overrides["experiment_name"] = args.experiment_name
    if args.dataset_root is not None:
        overrides["dataset_root"] = args.dataset_root

    split_overrides: dict[str, Any] = {}
    if args.train_ratio is not None:
        split_overrides["train_ratio"] = args.train_ratio
    if args.validation_ratio is not None:
        split_overrides["validation_ratio"] = args.validation_ratio
    if args.test_ratio is not None:
        split_overrides["test_ratio"] = args.test_ratio
    if args.stratified is not None:
        split_overrides["stratified"] = args.stratified
    if split_overrides:
        overrides["split"] = split_overrides

    training_overrides: dict[str, Any] = {}
    if args.epochs is not None:
        training_overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        training_overrides["batch_size"] = args.batch_size
    if args.early_stopping_patience is not None:
        training_overrides["early_stopping_patience"] = args.early_stopping_patience
    if args.early_stopping_min_delta is not None:
        training_overrides["early_stopping_min_delta"] = args.early_stopping_min_delta
    if args.learning_rate is not None:
        training_overrides["learning_rate"] = args.learning_rate
    if args.scheduler_factor is not None:
        training_overrides["scheduler_factor"] = args.scheduler_factor
    if args.scheduler_patience is not None:
        training_overrides["scheduler_patience"] = args.scheduler_patience
    if args.min_learning_rate is not None:
        training_overrides["min_learning_rate"] = args.min_learning_rate
    if args.augmentation_mode is not None:
        training_overrides["augmentation_mode"] = args.augmentation_mode
    if args.resume_from is not None:
        training_overrides["resume_from"] = args.resume_from
    if training_overrides:
        overrides["training"] = training_overrides

    model_overrides: dict[str, Any] = {}
    if args.model_name is not None:
        model_overrides["name"] = args.model_name

    model_options: dict[str, Any] = {}
    if args.use_pretrained is not None:
        model_options["use_pretrained"] = args.use_pretrained
    if args.freeze_backbone is not None:
        model_options["freeze_backbone"] = args.freeze_backbone
    if model_options:
        model_overrides["options"] = model_options

    if model_overrides:
        overrides["model"] = model_overrides

    return overrides or None


def _build_config_loader(
    defaults_path: str,
    split_seed_override: int | None,
    config_overrides: dict[str, Any] | None,
) -> JsonConfigLoader:
    """Builds config loader with optional split-seed override for reproducibility studies."""

    return JsonConfigLoader(
        defaults_path=defaults_path,
        split_seed_override=split_seed_override,
        config_overrides=config_overrides,
    )


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="EuroSAT project scaffold CLI")
    parser.add_argument(
        "--config",
        default="configs/baseline_cnn.json",
        help="Path to an experiment configuration file.",
    )
    parser.add_argument(
        "--defaults",
        default="configs/experiment.defaults.json",
        help="Path to canonical default configuration values.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional split seed override to run reproducibility experiments.",
    )
    parser.add_argument("--experiment-name", default=None, help="Optional experiment_name override.")
    parser.add_argument("--dataset-root", default=None, help="Optional dataset_root override.")
    parser.add_argument("--model-name", default=None, help="Optional model.name override.")
    parser.add_argument("--train-ratio", type=float, default=None, help="Optional split.train_ratio override.")
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=None,
        help="Optional split.validation_ratio override.",
    )
    parser.add_argument("--test-ratio", type=float, default=None, help="Optional split.test_ratio override.")
    parser.add_argument(
        "--stratified",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional split.stratified override.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Optional training.epochs override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional training.batch_size override.")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Optional training.early_stopping_patience override.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="Optional training.early_stopping_min_delta override.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional training.learning_rate override.",
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=None,
        help="Optional training.scheduler_factor override.",
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=None,
        help="Optional training.scheduler_patience override.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=None,
        help="Optional training.min_learning_rate override.",
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=["none", "flips", "full"],
        default=None,
        help="Optional training.augmentation_mode override.",
    )
    parser.add_argument("--resume-from", default=None, help="Optional training.resume_from override.")
    parser.add_argument(
        "--use-pretrained",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional model.options.use_pretrained override.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Optional model.options.freeze_backbone override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the scaffold and print the resolved experiment payload.",
    )
    parser.add_argument(
        "--prepare-dataset",
        action="store_true",
        help="Build dataset index, validate inputs, and persist deterministic split artifacts.",
    )
    parser.add_argument(
        "--splits-output",
        default="artifacts/splits",
        help="Output directory for split artifacts.",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Run baseline reference training and evaluation using saved split artifacts.",
    )
    parser.add_argument(
        "--reports-output",
        default="artifacts/reports/baseline_metrics.json",
        help="Output JSON file path for baseline metrics report.",
    )
    parser.add_argument(
        "--checkpoints-output",
        default="checkpoints/baseline",
        help="Output directory for baseline checkpoint artifacts.",
    )
    return parser


def main() -> int:
    """Runs the CLI entrypoint."""

    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    config_overrides = _build_config_overrides(args)
    config_loader = _build_config_loader(args.defaults, args.seed, config_overrides)

    if args.prepare_dataset:
        prepare_dataset_use_case = PrepareDataset(
            config_loader=config_loader,
            dataset_indexer=EuroSatDatasetIndexer(),
            dataset_splitter=StratifiedSplitter(),
            split_persistence=JsonSplitPersistence(),
        )
        payload = prepare_dataset_use_case.execute(
            config_path=args.config,
            output_dir=args.splits_output,
        )
        print(json.dumps(payload, indent=2))
        return 0

    if args.run_baseline:
        config = config_loader.load(args.config)

        split_dir = Path(args.splits_output)
        split_artifacts = {
            "train": (split_dir / "train_split.json").as_posix(),
            "validation": (split_dir / "validation_split.json").as_posix(),
            "test": (split_dir / "test_split.json").as_posix(),
        }

        orchestrator = TrainingOrchestrator(
            model_factory=SharedModelFactory(),
            data_loader_factory=SplitJsonLoaderFactory(),
            trainer=BaselineTrainer(),
            evaluator=BaselineEvaluator(list(EXPECTED_EUROSAT_CLASSES)),
            checkpoint_store=JsonCheckpointStore(),
            report_writer=JsonReportWriter(),
        )

        result = orchestrator.run(
            config=config,
            split_artifacts=split_artifacts,
            output_dir=args.checkpoints_output,
            report_output_path=args.reports_output,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.dry_run:
        dry_run_use_case = StartTraining(
            config_loader,
            DryRunTrainingRunner(),
        )
        print(dry_run_use_case.execute(args.config))
        return 0

    parser.print_help()
    return 0
