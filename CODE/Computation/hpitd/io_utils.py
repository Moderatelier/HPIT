"""UTF-8 logging and machine-readable status helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def configure_logging(log_path: Path) -> logging.Logger:
    """Create a console and UTF-8 file logger for one run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hpitd")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary_path.replace(path)


def write_status(path: Path, state: str, **details: Any) -> None:
    """Write a compact run-status record."""
    write_json(path, {"state": state, "updated_at": utc_timestamp(), **details})
