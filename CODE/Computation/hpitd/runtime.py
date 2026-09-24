"""Runtime selection and reproducibility helpers."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


def select_device(project_config: dict[str, Any], override: str | None = None) -> torch.device:
    """Resolve `auto`, `cpu`, `cuda`, or an explicit PyTorch device string."""
    requested = override or str(project_config["device"])
    if requested == "auto":
        requested = f"cuda:{int(project_config['gpu_id'])}" if torch.cuda.is_available() else "cpu"
    elif requested == "cuda":
        requested = f"cuda:{int(project_config['gpu_id'])}"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def set_reproducible_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trainable_parameter_count(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
