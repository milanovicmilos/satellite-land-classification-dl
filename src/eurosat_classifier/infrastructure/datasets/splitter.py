"""Deterministic stratified splitting for EuroSAT dataset indices."""

import random

from eurosat_classifier.domain.entities import DatasetIndex, DatasetSplit, LabeledSample, PreparedSplit


class StratifiedSplitter:
    """Creates deterministic train/validation/test splits per class."""

    def split(self, dataset_index: DatasetIndex, split: DatasetSplit) -> PreparedSplit:
        split.validate()
        if not split.stratified:
            raise ValueError("Only stratified splitting is supported in this phase.")

        train_samples: list[LabeledSample] = []
        validation_samples: list[LabeledSample] = []
        test_samples: list[LabeledSample] = []

        rng = random.Random(split.seed)

        for class_name in sorted(dataset_index.samples_by_class):
            class_samples = list(dataset_index.samples_by_class[class_name])
            if len(class_samples) < 3:
                raise ValueError(
                    f"Class '{class_name}' must have at least 3 samples for train/val/test splitting."
                )

            rng.shuffle(class_samples)

            train_count, validation_count, test_count = self._calculate_counts(
                len(class_samples),
                split.train_ratio,
                split.validation_ratio,
                class_name,
            )

            train_end = train_count
            validation_end = train_count + validation_count

            train_samples.extend(class_samples[:train_end])
            validation_samples.extend(class_samples[train_end:validation_end])
            test_samples.extend(class_samples[validation_end : validation_end + test_count])

        return PreparedSplit(
            train=train_samples,
            validation=validation_samples,
            test=test_samples,
            seed=split.seed,
        )

    @staticmethod
    def _calculate_counts(
        total_count: int,
        train_ratio: float,
        validation_ratio: float,
        class_name: str,
    ) -> tuple[int, int, int]:
        train_count = int(total_count * train_ratio)
        validation_count = int(total_count * validation_ratio)
        test_count = total_count - train_count - validation_count

        if train_count == 0 or validation_count == 0 or test_count == 0:
            raise ValueError(
                f"Split configuration produced an empty partition for class '{class_name}' "
                f"(total={total_count}, train={train_count}, validation={validation_count}, test={test_count})."
            )

        return train_count, validation_count, test_count
