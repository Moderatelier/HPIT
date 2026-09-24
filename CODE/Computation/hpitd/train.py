"""Training workflow for the released HPIT architecture."""

from __future__ import annotations

import logging
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import experiment_dir, resolve_project_path
from .data import PointCloudDataset, create_training_loaders, dataset_from_config
from .io_utils import utc_timestamp, write_json
from .metrics import mean_metric_records, regression_metrics
from .model import build_model
from .runtime import select_device, set_reproducible_seed, trainable_parameter_count


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_loss: float,
    best_epoch: int,
    history: dict[str, list[float]],
    dataset: PointCloudDataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    random_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        random_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    public_config = deepcopy(config)
    public_config.pop("_config_path", None)
    return {
        "format_version": 1,
        "created_at": utc_timestamp(),
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "history": history,
        "normalisation": dataset.statistics.to_dict(),
        "config": public_config,
        "random_state": random_state,
    }


def _restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> tuple[int, float, int, dict[str, list[float]]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    random_state = checkpoint.get("random_state", {})
    if "python" in random_state:
        random.setstate(random_state["python"])
    if "numpy" in random_state:
        np.random.set_state(random_state["numpy"])
    if "torch" in random_state:
        torch.set_rng_state(random_state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in random_state:
        torch.cuda.set_rng_state_all(random_state["torch_cuda"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_validation_loss"]),
        int(checkpoint["best_epoch"]),
        checkpoint["history"],
    )


def _validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    model.eval()
    losses: list[float] = []
    metric_records: list[dict[str, float]] = []
    with torch.no_grad():
        for coordinates, physical_features, targets in loader:
            coordinates = coordinates.to(device, non_blocking=True)
            physical_features = physical_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            predictions = model(coordinates, physical_features)
            losses.append(float(loss_function(predictions, targets).item()))
            metric_records.append(
                regression_metrics(
                    predictions.detach().cpu().numpy(), targets.detach().cpu().numpy()
                )
            )
    model.train()
    return float(np.mean(losses)), mean_metric_records(metric_records)


def train_model(
    config: dict[str, Any],
    logger: logging.Logger,
    run_name: str | None = None,
    device_override: str | None = None,
) -> dict[str, Any]:
    """Run training and return a machine-readable summary."""
    project_config = config["project"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    run_directory = experiment_dir(config, run_name)
    checkpoint_directory = run_directory / "Checkpoints"
    metrics_directory = run_directory / "Metrics"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    metrics_directory.mkdir(parents=True, exist_ok=True)

    seed = int(project_config["seed"])
    set_reproducible_seed(seed)
    device = select_device(project_config, device_override)
    dataset = dataset_from_config(resolve_project_path(data_config["train_dir"]), data_config)
    train_loader, validation_loader, split_manifest = create_training_loaders(
        dataset,
        data_config,
        training_config,
        seed=seed,
        use_pin_memory=device.type == "cuda",
    )
    write_json(run_directory / "split_manifest.json", split_manifest)
    write_json(run_directory / "normalisation.json", dataset.statistics.to_dict())

    model = build_model(model_config).to(device)
    parameter_count = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training_config["scheduler_t_max"])
    )
    loss_function = nn.SmoothL1Loss()

    resume_path = checkpoint_directory / "resume_checkpoint.pth"
    best_path = checkpoint_directory / "best_model.pth"
    history: dict[str, list[float]] = {
        "training_loss": [],
        "validation_loss": [],
        "learning_rate": [],
    }
    start_epoch = 0
    best_validation_loss = float("inf")
    best_epoch = 0
    if bool(training_config["resume"]) and resume_path.is_file():
        start_epoch, best_validation_loss, best_epoch, history = _restore_checkpoint(
            resume_path, model, optimizer, scheduler, device
        )
        logger.info("Resumed training from epoch %d", start_epoch + 1)

    logger.info("Device: %s", device)
    logger.info("Training samples: %d", len(split_manifest["training"]))
    logger.info("Validation samples: %d", len(split_manifest["validation"]))
    logger.info("Trainable parameters: %d", parameter_count)
    model.train()
    started_at = time.time()
    epoch_count = int(training_config["epochs"])
    validation_frequency = int(training_config["validation_frequency"])
    checkpoint_frequency = int(training_config["checkpoint_frequency"])
    best_model_start_epoch = int(training_config["best_model_start_epoch"])
    max_gradient_norm = float(training_config["max_gradient_norm"])

    for epoch in range(start_epoch, epoch_count):
        epoch_started_at = time.time()
        training_losses: list[float] = []
        for coordinates, physical_features, targets in train_loader:
            coordinates = coordinates.to(device, non_blocking=True)
            physical_features = physical_features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(coordinates, physical_features)
            loss = loss_function(predictions, targets)
            loss.backward()
            if max_gradient_norm >= 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            optimizer.step()
            training_losses.append(float(loss.item()))

        scheduler.step()
        mean_training_loss = float(np.mean(training_losses))
        history["training_loss"].append(mean_training_loss)
        history["learning_rate"].append(float(optimizer.param_groups[0]["lr"]))
        logger.info(
            "Epoch %d/%d | training loss %.8f | lr %.8g | %.2f s",
            epoch + 1,
            epoch_count,
            mean_training_loss,
            optimizer.param_groups[0]["lr"],
            time.time() - epoch_started_at,
        )

        if (epoch + 1) % validation_frequency == 0:
            validation_loss, validation_metrics = _validate(
                model, validation_loader, loss_function, device
            )
            history["validation_loss"].append(validation_loss)
            logger.info(
                "Validation | loss %.8f | RMSE %.8f | MAE %.8f | R2 %.8f",
                validation_loss,
                validation_metrics["rmse"],
                validation_metrics["mae"],
                validation_metrics["r2"],
            )
            if epoch + 1 >= best_model_start_epoch and validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                torch.save(
                    _checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        best_validation_loss,
                        best_epoch,
                        history,
                        dataset,
                        config,
                    ),
                    best_path,
                )
                logger.info("Saved best model at epoch %d", best_epoch)

        if (epoch + 1) % checkpoint_frequency == 0:
            torch.save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_validation_loss,
                    best_epoch,
                    history,
                    dataset,
                    config,
                ),
                resume_path,
            )
            write_json(metrics_directory / "history.json", history)

    final_epoch = max(start_epoch, epoch_count) - 1
    final_payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        final_epoch,
        best_validation_loss,
        best_epoch,
        history,
        dataset,
        config,
    )
    torch.save(final_payload, resume_path)
    if not best_path.exists():
        torch.save(final_payload, best_path)
        best_epoch = epoch_count
        best_validation_loss = (
            history["validation_loss"][-1] if history["validation_loss"] else float("inf")
        )
    write_json(metrics_directory / "history.json", history)
    summary = {
        "state": "completed",
        "device": str(device),
        "epochs_completed": epoch_count,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "trainable_parameters": parameter_count,
        "elapsed_seconds": time.time() - started_at,
        "best_checkpoint": str(best_path),
        "resume_checkpoint": str(resume_path),
    }
    write_json(run_directory / "training_summary.json", summary)
    return summary
