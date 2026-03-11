"""Application use cases for training orchestration."""

from eurosat_classifier.application.contracts import ConfigLoader, TrainingRunner


class StartTraining:
    """Coordinates configuration loading and training execution."""

    def __init__(self, config_loader: ConfigLoader, training_runner: TrainingRunner) -> None:
        self._config_loader = config_loader
        self._training_runner = training_runner

    def execute(self, config_path: str) -> str:
        config = self._config_loader.load(config_path)
        experiment = config.to_experiment()
        return self._training_runner.run(experiment)
