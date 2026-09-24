"""Checkpoint evaluation on independent point-cloud samples."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import experiment_dir, resolve_project_path
from .data import NormalisationStatistics, create_evaluation_loader, dataset_from_config
from .io_utils import write_json
from .metrics import mean_metric_records, regression_metrics
from .model import build_model
from .runtime import select_device, trainable_parameter_count


def evaluate_model(
    config: dict[str, Any],
    logger: logging.Logger,
    checkpoint_path: Path | None = None,
    run_name: str | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint and save aggregate and per-sample metrics."""
    run_directory = experiment_dir(config, run_name)
    checkpoint_path = checkpoint_path or run_directory / "Checkpoints" / "best_model.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    device = select_device(config["project"], device_override)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "normalisation" not in checkpoint:
        raise ValueError("Checkpoint does not contain released-format normalisation metadata")

    checkpoint_config = checkpoint.get("config", config)
    model = build_model(checkpoint_config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    statistics = NormalisationStatistics.from_dict(checkpoint["normalisation"])
    data_config = checkpoint_config["data"]
    dataset = dataset_from_config(
        resolve_project_path(config["data"]["test_dir"]),
        data_config,
        statistics=statistics,
    )
    loader = create_evaluation_loader(
        dataset, data_config, use_pin_memory=device.type == "cuda"
    )
    evaluation_directory = run_directory / "Evaluation"
    prediction_directory = evaluation_directory / "Predictions"
    save_predictions = bool(config["evaluation"]["save_predictions"])
    if save_predictions:
        prediction_directory.mkdir(parents=True, exist_ok=True)

    per_sample: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, (coordinates, physical_features, normalised_targets) in enumerate(loader):
            coordinates = coordinates.to(device, non_blocking=True)
            physical_features = physical_features.to(device, non_blocking=True)
            normalised_targets = normalised_targets.to(device, non_blocking=True)
            normalised_predictions = model(coordinates, physical_features)
            predictions = dataset.inverse_target(normalised_predictions)
            targets = dataset.inverse_target(normalised_targets)
            prediction_array = predictions.detach().cpu().numpy()
            target_array = targets.detach().cpu().numpy()
            metrics = regression_metrics(
                prediction_array,
                target_array,
                relative_error_epsilon=float(
                    config["evaluation"]["relative_error_epsilon"]
                ),
            )
            record: dict[str, Any] = {"sample": dataset.sample_names[index], **metrics}
            per_sample.append(record)
            logger.info(
                "%s | RMSE %.8f | MAE %.8f | R2 %.8f | mean relative error %.8f",
                dataset.sample_names[index],
                metrics["rmse"],
                metrics["mae"],
                metrics["r2"],
                metrics["mean_relative_error"],
            )
            if save_predictions:
                np.savez_compressed(
                    prediction_directory / f"{Path(dataset.sample_names[index]).stem}.npz",
                    prediction=prediction_array[0],
                    target=target_array[0],
                )

    aggregate = mean_metric_records(
        [{key: value for key, value in record.items() if key != "sample"} for record in per_sample]
    )
    summary = {
        "state": "completed",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
        "sample_count": len(per_sample),
        "aggregate_metrics": aggregate,
        "per_sample_metrics": per_sample,
    }
    write_json(evaluation_directory / "evaluation_summary.json", summary)
    return summary
