"""Lazy loader for a user-supplied, patched CRA-PCN checkout."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import PROJECT_ROOT


CRA_PCN_COMMIT = "e87d11a5134332366c507f7cf784a03ad13a4d00"
DEFAULT_CRA_PCN_ROOT = PROJECT_ROOT / "third_party" / "CRA-PCN"


class CRACompatibilityError(RuntimeError):
    """Raised when the external CRA-PCN dependency is absent or unpatched."""


def cra_pcn_root() -> Path:
    """Return the configured external CRA-PCN checkout path."""
    value = os.environ.get("HPITD_CRA_PCN_ROOT")
    return Path(value).expanduser().resolve() if value else DEFAULT_CRA_PCN_ROOT.resolve()


@lru_cache(maxsize=1)
def load_cra_pcn_module() -> ModuleType:
    """Load the patched upstream module without distributing its source."""
    root = cra_pcn_root()
    module_path = root / "models" / "crapcn.py"
    if not module_path.is_file():
        raise CRACompatibilityError(
            "CRA-PCN is required but was not found. Follow docs/cra_pcn_setup.md, "
            f"or set HPITD_CRA_PCN_ROOT. Expected: {module_path}"
        )
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("models.crapcn")
    loaded_path = Path(module.__file__).resolve()
    if root not in loaded_path.parents:
        raise CRACompatibilityError(
            f"Python imported models.crapcn from an unexpected location: {loaded_path}"
        )
    if getattr(module, "HPITD_COMPATIBILITY_PATCH", None) != 1:
        raise CRACompatibilityError(
            "The CRA-PCN checkout does not contain the HPITD compatibility patch. "
            "Apply patches/cra-pcn-hpit-compatibility.patch."
        )
    forward_parameters = inspect.signature(module.CRT.forward).parameters
    if "num_groups" not in forward_parameters:
        raise CRACompatibilityError("The patched CRT.forward method must accept num_groups")
    return module


def create_cross_relation_transformer(**kwargs: Any):
    """Construct the patched upstream CRT class."""
    return load_cra_pcn_module().CRT(**kwargs)


def hierarchical_fps(*args: Any, **kwargs: Any):
    """Call the patched upstream hierarchical FPS helper."""
    return load_cra_pcn_module().hierarchical_fps(*args, **kwargs)
