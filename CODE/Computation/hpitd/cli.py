"""Command-line interface for training, evaluation, and self-checks."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from .config import (
    DEFAULT_CONFIG_PATH,
    experiment_dir,
    load_config,
    resolve_project_path,
    resolved_config,
)
from .evaluate import evaluate_model
from .io_utils import configure_logging, write_status
from .model import build_model
from .point_ops import backend_name
from .runtime import select_device, trainable_parameter_count
from .train import train_model


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON configuration file.",
    )
    parser.add_argument("--run-name", default=None, help="Experiment directory name override.")
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime device override, for example cpu, cuda, or cuda:0.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hierarchical physics-integrated thermal-deformation model"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model.")
    _add_common_arguments(train_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint.")
    _add_common_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Checkpoint path override."
    )

    self_test_parser = subparsers.add_parser(
        "self-test", help="Run a lightweight model-forward smoke test."
    )
    self_test_parser.add_argument("--device", default="cpu")

    config_parser = subparsers.add_parser(
        "show-config", help="Print the resolved default or supplied configuration."
    )
    config_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def run_self_test(device_name: str) -> dict[str, Any]:
    project_config = {"device": device_name, "gpu_id": 0}
    device = select_device(project_config)
    model_config = {
        "space_dim": 3,
        "feature_dim": 6,
        "output_dim": 1,
        "num_layers": 1,
        "hidden_dim": 16,
        "dropout": 0.0,
        "num_heads": 4,
        "activation": "gelu",
        "mlp_ratio": 1,
        "knn_k": 4,
        "slice_count": 16,
        "downsample_rates": [1, 4],
    }
    model = build_model(model_config).to(device).eval()
    coordinates = torch.rand(1, 32, 3, device=device)
    physical_features = torch.rand(1, 32, 6, device=device)
    with torch.no_grad():
        output = model(coordinates, physical_features)
    expected_shape = (1, 32, 1)
    if tuple(output.shape) != expected_shape or not torch.isfinite(output).all():
        raise RuntimeError(
            f"Self-test failed: expected finite output {expected_shape}, received {output.shape}"
        )
    return {
        "state": "passed",
        "device": str(device),
        "point_backend": backend_name(coordinates),
        "output_shape": list(output.shape),
        "trainable_parameters": trainable_parameter_count(model),
    }


def _run_managed_command(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    run_directory = experiment_dir(config, args.run_name)
    log_directory = run_directory / "Logs"
    status_path = run_directory / "run_status.json"
    logger = configure_logging(log_directory / f"{args.command}.log")
    write_status(status_path, "running", command=args.command)
    try:
        if args.command == "train":
            summary = train_model(config, logger, args.run_name, args.device)
        else:
            checkpoint = args.checkpoint
            if checkpoint is not None and not checkpoint.is_absolute():
                checkpoint = resolve_project_path(checkpoint)
            summary = evaluate_model(
                config,
                logger,
                checkpoint_path=checkpoint,
                run_name=args.run_name,
                device_override=args.device,
            )
        write_status(status_path, "completed", command=args.command, summary=summary)
        return summary
    except Exception as error:
        logger.error("%s", traceback.format_exc())
        write_status(status_path, "failed", command=args.command, error=str(error))
        raise


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        if args.command == "self-test":
            result = run_self_test(args.device)
        elif args.command == "show-config":
            result = resolved_config(load_config(args.config))
        else:
            config = load_config(args.config)
            result = _run_managed_command(args, config)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
