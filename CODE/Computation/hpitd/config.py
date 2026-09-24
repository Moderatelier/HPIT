"""Configuration loading and project-relative path handling."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
COMPUTATION_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = COMPUTATION_DIR.parents[1]
DEFAULT_CONFIG_PATH = COMPUTATION_DIR / "Config" / "default.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and minimally validate a JSON configuration file."""
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    required_sections = {"project", "data", "model", "training", "evaluation"}
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")

    model = config["model"]
    if model["hidden_dim"] % model["num_heads"] != 0:
        raise ValueError("model.hidden_dim must be divisible by model.num_heads")
    if len(model["downsample_rates"]) != 2:
        raise ValueError("The released architecture expects two down-sampling rates")
    if not 0.0 < config["data"]["train_fraction"] < 1.0:
        raise ValueError("data.train_fraction must be between zero and one")

    config["_config_path"] = str(config_path)
    return config


def resolved_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a serialisable copy with project-relative paths resolved."""
    result = deepcopy(config)
    result.pop("_config_path", None)
    for key in ("train_dir", "test_dir"):
        result["data"][key] = str(resolve_project_path(result["data"][key]))
    result["project"]["output_root"] = str(
        resolve_project_path(result["project"]["output_root"])
    )
    return result


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def experiment_dir(config: dict[str, Any], run_name: str | None = None) -> Path:
    """Return the experiment output directory."""
    name = run_name or config["project"]["run_name"]
    if not name or any(part in name for part in ("/", "\\", "..")):
        raise ValueError("run_name must be a single clear directory name")
    return resolve_project_path(config["project"]["output_root"]) / name
