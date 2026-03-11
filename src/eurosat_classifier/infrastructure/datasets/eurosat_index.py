"""EuroSAT dataset indexing and validation services."""

from pathlib import Path

from eurosat_classifier.domain.entities import DatasetIndex, LabeledSample


EXPECTED_EUROSAT_CLASSES: tuple[str, ...] = (
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
)

ALLOWED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg")


class EuroSatDatasetIndexer:
    """Builds a validated index for the EuroSAT RGB JPEG dataset."""

    def __init__(
        self,
        expected_classes: tuple[str, ...] = EXPECTED_EUROSAT_CLASSES,
        allowed_extensions: tuple[str, ...] = ALLOWED_EXTENSIONS,
    ) -> None:
        self._expected_classes = expected_classes
        self._allowed_extensions = tuple(ext.lower() for ext in allowed_extensions)

    def build(self, dataset_root: str) -> DatasetIndex:
        root_path = Path(dataset_root)
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(f"Dataset root does not exist or is not a directory: {dataset_root}")

        discovered_classes = sorted(path.name for path in root_path.iterdir() if path.is_dir())
        self._validate_class_structure(discovered_classes)

        samples_by_class: dict[str, list[LabeledSample]] = {}

        for class_index, class_name in enumerate(self._expected_classes):
            class_dir = root_path / class_name
            class_samples = self._collect_class_samples(class_dir, class_name, class_index)
            samples_by_class[class_name] = class_samples

        return DatasetIndex(dataset_root=str(root_path.as_posix()), samples_by_class=samples_by_class)

    def _validate_class_structure(self, discovered_classes: list[str]) -> None:
        expected_set = set(self._expected_classes)
        discovered_set = set(discovered_classes)

        missing_classes = sorted(expected_set - discovered_set)
        unexpected_classes = sorted(discovered_set - expected_set)

        if missing_classes:
            raise ValueError(f"Dataset is missing expected class folders: {missing_classes}")

        if unexpected_classes:
            raise ValueError(f"Dataset contains unexpected class folders: {unexpected_classes}")

    def _collect_class_samples(
        self,
        class_dir: Path,
        class_name: str,
        class_index: int,
    ) -> list[LabeledSample]:
        files = sorted(path for path in class_dir.iterdir() if path.is_file())
        if not files:
            raise ValueError(f"Class folder has no image files: {class_dir.as_posix()}")

        samples: list[LabeledSample] = []
        for file_path in files:
            extension = file_path.suffix.lower()
            if extension not in self._allowed_extensions:
                raise ValueError(
                    f"Unsupported extension '{extension}' in {file_path.as_posix()}. "
                    f"Allowed: {self._allowed_extensions}"
                )

            if not self._is_valid_jpeg(file_path):
                raise ValueError(f"Corrupt or invalid JPEG file: {file_path.as_posix()}")

            samples.append(
                LabeledSample(
                    path=file_path.as_posix(),
                    class_name=class_name,
                    class_index=class_index,
                )
            )
        return samples

    @staticmethod
    def _is_valid_jpeg(file_path: Path) -> bool:
        """Performs a lightweight JPEG validity check without third-party dependencies."""

        try:
            with file_path.open("rb") as handle:
                header = handle.read(16)
                handle.seek(-2, 2)
                footer = handle.read(2)
        except OSError:
            return False

        if len(header) < 4:
            return False

        has_soi = header[:2] == b"\xff\xd8"
        has_eoi = footer == b"\xff\xd9"
        return has_soi and has_eoi
