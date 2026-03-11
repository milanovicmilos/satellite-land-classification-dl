"""Tests for deterministic stratified splitting and split persistence."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.domain.entities import DatasetIndex, DatasetSplit, LabeledSample
from eurosat_classifier.infrastructure.datasets.split_store import JsonSplitPersistence
from eurosat_classifier.infrastructure.datasets.splitter import StratifiedSplitter


class StratifiedSplitterTests(unittest.TestCase):
    def _build_index(self, samples_per_class: int = 20) -> DatasetIndex:
        classes = [
            "AnnualCrop",
            "Forest",
            "HerbaceousVegetation",
            "Highway",
            "Industrial",
            "Pasture",
            "PermanentCrop",
            "Residential",
            "River",
            "SeaLake",
        ]
        by_class: dict[str, list[LabeledSample]] = {}

        for class_index, class_name in enumerate(classes):
            samples = []
            for sample_idx in range(samples_per_class):
                samples.append(
                    LabeledSample(
                        path=f"data/EuroSAT/{class_name}/sample_{sample_idx}.jpg",
                        class_name=class_name,
                        class_index=class_index,
                    )
                )
            by_class[class_name] = samples

        return DatasetIndex(dataset_root="data/EuroSAT", samples_by_class=by_class)

    def test_split_is_deterministic_for_same_seed(self) -> None:
        dataset_index = self._build_index()
        config = DatasetSplit(0.7, 0.15, 0.15, seed=42, stratified=True)
        splitter = StratifiedSplitter()

        first = splitter.split(dataset_index, config)
        second = splitter.split(dataset_index, config)

        self.assertEqual([s.path for s in first.train], [s.path for s in second.train])
        self.assertEqual([s.path for s in first.validation], [s.path for s in second.validation])
        self.assertEqual([s.path for s in first.test], [s.path for s in second.test])

    def test_split_counts_match_expected_ratios(self) -> None:
        dataset_index = self._build_index(samples_per_class=20)
        config = DatasetSplit(0.7, 0.15, 0.15, seed=42, stratified=True)
        splitter = StratifiedSplitter()

        prepared = splitter.split(dataset_index, config)

        self.assertEqual(len(prepared.train), 140)
        self.assertEqual(len(prepared.validation), 30)
        self.assertEqual(len(prepared.test), 30)

    def test_split_persistence_writes_expected_files(self) -> None:
        dataset_index = self._build_index(samples_per_class=5)
        config = DatasetSplit(0.6, 0.2, 0.2, seed=7, stratified=True)
        splitter = StratifiedSplitter()
        persistence = JsonSplitPersistence()

        prepared = splitter.split(dataset_index, config)

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            output = persistence.save(prepared, tmp_dir.as_posix())
            self.assertTrue(Path(output["train"]).exists())
            self.assertTrue(Path(output["validation"]).exists())
            self.assertTrue(Path(output["test"]).exists())
            self.assertTrue(Path(output["manifest"]).exists())
        finally:
            shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    unittest.main()
