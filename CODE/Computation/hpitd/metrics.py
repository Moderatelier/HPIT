"""Regression metrics used during validation and evaluation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    relative_error_epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compute scalar regression metrics on flattened arrays."""
    prediction_flat = np.asarray(prediction, dtype=np.float64).reshape(-1)
    target_flat = np.asarray(target, dtype=np.float64).reshape(-1)
    if prediction_flat.shape != target_flat.shape:
        raise ValueError("Prediction and target shapes must match")
    if prediction_flat.size == 0:
        raise ValueError("Metric arrays must not be empty")

    residual = prediction_flat - target_flat
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    target_variation = float(np.sum((target_flat - np.mean(target_flat)) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / target_variation) if target_variation > 0 else 0.0
    relative_error = np.abs(residual) / np.maximum(np.abs(target_flat), relative_error_epsilon)
    return {
        "rmse": math.sqrt(mse),
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "mean_relative_error": float(np.mean(relative_error)),
        "max_relative_error": float(np.max(relative_error)),
        "min_relative_error": float(np.min(relative_error)),
    }


def mean_metric_records(records: Iterable[dict[str, float]]) -> dict[str, float]:
    """Average matching scalar fields across metric records."""
    record_list = list(records)
    if not record_list:
        raise ValueError("At least one metric record is required")
    keys = record_list[0].keys()
    return {key: float(np.mean([record[key] for record in record_list])) for key in keys}
