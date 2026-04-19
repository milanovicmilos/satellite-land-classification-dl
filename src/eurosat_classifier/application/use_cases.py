"""Application use cases for training orchestration."""

from eurosat_classifier.application.contracts import (
    ConfigLoader,
    DatasetIndexer,
    DatasetSplitter,
    SplitPersistence,
    TrainingRunner,
)


class StartTraining:
    """Coordinates configuration loading and training execution."""

    def __init__(self, config_loader: ConfigLoader, training_runner: TrainingRunner) -> None:
        self._config_loader = config_loader
        self._training_runner = training_runner

    def execute(self, config_path: str) -> str:
        config = self._config_loader.load(config_path)
        experiment = config.to_experiment()
        return self._training_runner.run(experiment)


class PrepareDataset:
    """Indexes dataset data and writes deterministic split artifacts."""

    def __init__(
        self,
        config_loader: ConfigLoader,
        dataset_indexer: DatasetIndexer,
        dataset_splitter: DatasetSplitter,
        split_persistence: SplitPersistence,
    ) -> None:
        self._config_loader = config_loader
        self._dataset_indexer = dataset_indexer
        self._dataset_splitter = dataset_splitter
        self._split_persistence = split_persistence

    def execute(self, config_path: str, output_dir: str) -> dict[str, object]:
        config = self._config_loader.load(config_path)
        experiment = config.to_experiment()

        dataset_index = self._dataset_indexer.build(experiment.dataset_root)
        prepared_split = self._dataset_splitter.split(dataset_index, experiment.split)
        artifacts = self._split_persistence.save(prepared_split, output_dir)

        return {
            "dataset_root": experiment.dataset_root,
            "seed": prepared_split.seed,
            "class_count": dataset_index.total_classes(),
            "total_samples": dataset_index.total_samples(),
            "train_samples": len(prepared_split.train),
            "validation_samples": len(prepared_split.validation),
            "test_samples": len(prepared_split.test),
            "artifacts": artifacts,
        }
