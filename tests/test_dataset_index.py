"""Tests for EuroSAT dataset indexing and validation rules."""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_classifier.infrastructure.datasets.eurosat_index import (
    EXPECTED_EUROSAT_CLASSES,
    EuroSatDatasetIndexer,
)


def _valid_jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff\xda\x00\x08TEST-DATA\xff\xd9"


class EuroSatDatasetIndexerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_dir = Path(tempfile.mkdtemp())
        self.dataset_root = self._tmp_dir / "EuroSAT"
        self.dataset_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp_dir)

    def _build_valid_dataset(self) -> None:
        for class_name in EXPECTED_EUROSAT_CLASSES:
            class_dir = self.dataset_root / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            (class_dir / "sample.jpg").write_bytes(_valid_jpeg_bytes())

    def test_build_creates_index_for_valid_dataset(self) -> None:
        self._build_valid_dataset()
        indexer = EuroSatDatasetIndexer()

        index = indexer.build(self.dataset_root.as_posix())

        self.assertEqual(index.total_classes(), 10)
        self.assertEqual(index.total_samples(), 10)

    def test_build_raises_for_missing_class_folder(self) -> None:
        self._build_valid_dataset()
        shutil.rmtree(self.dataset_root / "SeaLake")

        indexer = EuroSatDatasetIndexer()

        with self.assertRaises(ValueError):
            indexer.build(self.dataset_root.as_posix())

    def test_build_raises_for_unsupported_extension(self) -> None:
        self._build_valid_dataset()
        (self.dataset_root / "Forest" / "bad.png").write_bytes(b"not-jpeg")

        indexer = EuroSatDatasetIndexer()

        with self.assertRaises(ValueError):
            indexer.build(self.dataset_root.as_posix())

    def test_build_raises_for_corrupt_jpeg(self) -> None:
        self._build_valid_dataset()
        (self.dataset_root / "Forest" / "broken.jpg").write_bytes(b"broken-jpeg")

        indexer = EuroSatDatasetIndexer()

        with self.assertRaises(ValueError):
            indexer.build(self.dataset_root.as_posix())


if __name__ == "__main__":
    unittest.main()
