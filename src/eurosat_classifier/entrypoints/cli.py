"""CLI entrypoint for scaffold verification and future orchestration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _build_config_loader(defaults_path: str, split_seed_override: int | None) -> JsonConfigLoader:
    """Builds config loader with optional split-seed override for reproducibility studies."""

    return JsonConfigLoader(defaults_path=defaults_path, split_seed_override=split_seed_override)


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="EuroSAT project scaffold CLI")
    parser.add_argument(
        "--config",
        default="configs/baseline_cnn_full.json",
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
    config_loader = _build_config_loader(args.defaults, args.seed)

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
