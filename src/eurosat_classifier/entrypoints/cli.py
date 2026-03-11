"""CLI entrypoint for scaffold verification and future orchestration."""

from __future__ import annotations

import argparse
import json

from eurosat_classifier.application.use_cases import PrepareDataset, StartTraining
from eurosat_classifier.infrastructure.config_loader import JsonConfigLoader
from eurosat_classifier.infrastructure.datasets.eurosat_index import EuroSatDatasetIndexer
from eurosat_classifier.infrastructure.datasets.split_store import JsonSplitPersistence
from eurosat_classifier.infrastructure.datasets.splitter import StratifiedSplitter
from eurosat_classifier.infrastructure.logging import configure_logging


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


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="EuroSAT project scaffold CLI")
    parser.add_argument(
        "--config",
        default="configs/baseline.example.json",
        help="Path to an experiment configuration file.",
    )
    parser.add_argument(
        "--defaults",
        default="configs/experiment.defaults.json",
        help="Path to canonical default configuration values.",
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
    return parser


def main() -> int:
    """Runs the CLI entrypoint."""

    configure_logging()
    parser = build_parser()
    args = parser.parse_args()

    if args.prepare_dataset:
        use_case = PrepareDataset(
            config_loader=JsonConfigLoader(defaults_path=args.defaults),
            dataset_indexer=EuroSatDatasetIndexer(),
            dataset_splitter=StratifiedSplitter(),
            split_persistence=JsonSplitPersistence(),
        )
        payload = use_case.execute(config_path=args.config, output_dir=args.splits_output)
        print(json.dumps(payload, indent=2))
        return 0

    if args.dry_run:
        use_case = StartTraining(
            JsonConfigLoader(defaults_path=args.defaults),
            DryRunTrainingRunner(),
        )
        print(use_case.execute(args.config))
        return 0

    parser.print_help()
    return 0
