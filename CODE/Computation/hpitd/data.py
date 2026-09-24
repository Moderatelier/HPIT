"""Point-cloud dataset loading and normalisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class NormalisationStatistics:
    """Statistics required to reproduce training transformations."""

    coordinate_mean: list[float]
    coordinate_std: list[float]
    feature_mean: list[float]
    feature_std: list[float]
    log_target_mean: float
    log_target_std: float
    target_floor: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NormalisationStatistics":
        return cls(**value)


def _safe_standard_deviation(values: np.ndarray, name: str) -> np.ndarray:
    standard_deviation = np.std(values, axis=0)
    if np.any(~np.isfinite(standard_deviation)):
        raise ValueError(f"Non-finite standard deviation found in {name}")
    return np.where(standard_deviation > 0.0, standard_deviation, 1.0)


class PointCloudDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Load `N x 17` NumPy point-cloud samples."""

    def __init__(
        self,
        data_dir: Path,
        coordinate_columns: list[int],
        feature_columns: list[int],
        target_column: int,
        target_floor: float,
        statistics: NormalisationStatistics | None = None,
    ) -> None:
        self.data_dir = data_dir
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        self.data_files = sorted(data_dir.glob("*.npy"))
        if not self.data_files:
            raise FileNotFoundError(f"No .npy samples found in: {data_dir}")

        self.coordinate_columns = coordinate_columns
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.target_floor = target_floor
        required_column_count = max(coordinate_columns + feature_columns + [target_column]) + 1
        self.point_clouds: list[np.ndarray] = []

        for path in self.data_files:
            point_cloud = np.load(path, allow_pickle=False)
            if point_cloud.ndim != 2 or point_cloud.shape[1] < required_column_count:
                raise ValueError(
                    f"{path.name} must be a two-dimensional array with at least "
                    f"{required_column_count} columns; received {point_cloud.shape}"
                )
            if not np.issubdtype(point_cloud.dtype, np.number):
                raise TypeError(f"{path.name} must contain numeric values")
            selected = point_cloud[:, coordinate_columns + feature_columns + [target_column]]
            if not np.all(np.isfinite(selected)):
                raise ValueError(f"Non-finite values found in {path.name}")
            self.point_clouds.append(np.asarray(point_cloud, dtype=np.float64))

        self.statistics = statistics or self._fit_statistics()

    @property
    def sample_names(self) -> list[str]:
        return [path.name for path in self.data_files]

    def _fit_statistics(self) -> NormalisationStatistics:
        coordinates = np.concatenate(
            [sample[:, self.coordinate_columns] for sample in self.point_clouds], axis=0
        )
        features = np.concatenate(
            [sample[:, self.feature_columns] for sample in self.point_clouds], axis=0
        )
        targets = np.concatenate(
            [sample[:, self.target_column] for sample in self.point_clouds], axis=0
        )
        positive_targets = np.maximum(targets, self.target_floor)
        log_targets = np.log(positive_targets)
        log_target_std = float(np.std(log_targets))
        if not np.isfinite(log_target_std):
            raise ValueError("The target standard deviation is non-finite")
        if log_target_std == 0.0:
            log_target_std = 1.0
        return NormalisationStatistics(
            coordinate_mean=np.mean(coordinates, axis=0).tolist(),
            coordinate_std=_safe_standard_deviation(coordinates, "coordinates").tolist(),
            feature_mean=np.mean(features, axis=0).tolist(),
            feature_std=_safe_standard_deviation(features, "features").tolist(),
            log_target_mean=float(np.mean(log_targets)),
            log_target_std=log_target_std,
            target_floor=float(self.target_floor),
        )

    def __len__(self) -> int:
        return len(self.point_clouds)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        point_cloud = self.point_clouds[index]
        statistics = self.statistics
        coordinate_mean = np.asarray(statistics.coordinate_mean)
        coordinate_std = np.asarray(statistics.coordinate_std)
        feature_mean = np.asarray(statistics.feature_mean)
        feature_std = np.asarray(statistics.feature_std)

        coordinates = (
            point_cloud[:, self.coordinate_columns] - coordinate_mean
        ) / coordinate_std
        features = (point_cloud[:, self.feature_columns] - feature_mean) / feature_std
        targets = np.maximum(point_cloud[:, self.target_column], statistics.target_floor)
        normalised_targets = (
            np.log(targets) - statistics.log_target_mean
        ) / statistics.log_target_std
        normalised_targets = normalised_targets[:, None]
        return (
            torch.from_numpy(coordinates).float(),
            torch.from_numpy(features).float(),
            torch.from_numpy(normalised_targets).float(),
        )

    def inverse_target(self, normalised_target: torch.Tensor) -> torch.Tensor:
        """Map a normalised logarithmic target back to the physical scale."""
        return torch.exp(
            normalised_target * self.statistics.log_target_std
            + self.statistics.log_target_mean
        )


def dataset_from_config(
    data_dir: Path,
    data_config: dict[str, Any],
    statistics: NormalisationStatistics | None = None,
) -> PointCloudDataset:
    """Construct a dataset from the shared data configuration."""
    return PointCloudDataset(
        data_dir=data_dir,
        coordinate_columns=[int(value) for value in data_config["coordinate_columns"]],
        feature_columns=[int(value) for value in data_config["feature_columns"]],
        target_column=int(data_config["target_column"]),
        target_floor=float(data_config["target_floor"]),
        statistics=statistics,
    )


def create_training_loaders(
    dataset: PointCloudDataset,
    data_config: dict[str, Any],
    training_config: dict[str, Any],
    seed: int,
    use_pin_memory: bool,
) -> tuple[DataLoader, DataLoader, dict[str, list[str]]]:
    """Create deterministic training and validation loaders."""
    train_size = int(float(data_config["train_fraction"]) * len(dataset))
    validation_size = len(dataset) - train_size
    if train_size < 1 or validation_size < 1:
        raise ValueError("At least two samples are required for the configured split")
    generator = torch.Generator().manual_seed(seed)
    train_subset, validation_subset = random_split(
        dataset, [train_size, validation_size], generator=generator
    )
    loader_options = {
        "num_workers": int(data_config["num_workers"]),
        "pin_memory": use_pin_memory,
    }
    train_loader = DataLoader(
        train_subset,
        batch_size=int(training_config["batch_size_train"]),
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=int(training_config["batch_size_validation"]),
        shuffle=False,
        **loader_options,
    )
    manifest = {
        "training": [dataset.sample_names[index] for index in sorted(train_subset.indices)],
        "validation": [
            dataset.sample_names[index] for index in sorted(validation_subset.indices)
        ],
    }
    return train_loader, validation_loader, manifest


def create_evaluation_loader(
    dataset: PointCloudDataset,
    data_config: dict[str, Any],
    use_pin_memory: bool,
) -> DataLoader:
    """Create an ordered single-sample evaluation loader."""
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=use_pin_memory,
    )
