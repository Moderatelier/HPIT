"""Lightweight tests that do not require research data."""

from __future__ import annotations

import sys
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPUTATION_DIR = PROJECT_ROOT / "CODE" / "Computation"
sys.path.insert(0, str(COMPUTATION_DIR))

from hpitd.config import load_config  # noqa: E402
from hpitd.cra_pcn_backend import (  # noqa: E402
    CRACompatibilityError,
    load_cra_pcn_module,
)
from hpitd.model import HierarchicalPhysicsIntegratedTransformer  # noqa: E402


class SmokeTests(unittest.TestCase):
    def test_default_configuration_loads(self) -> None:
        config = load_config()
        self.assertEqual(config["model"]["output_dim"], 1)
        self.assertEqual(config["data"]["target_column"], 16)

    def test_missing_external_dependency_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with patch.dict(os.environ, {"HPITD_CRA_PCN_ROOT": temporary_directory}):
                load_cra_pcn_module.cache_clear()
                with self.assertRaisesRegex(CRACompatibilityError, "cra_pcn_setup.md"):
                    load_cra_pcn_module()
        load_cra_pcn_module.cache_clear()

    def test_original_feature_offset_key_is_supported(self) -> None:
        model = HierarchicalPhysicsIntegratedTransformer.__new__(
            HierarchicalPhysicsIntegratedTransformer
        )
        torch.nn.Module.__init__(model)
        model.feature_offset = torch.nn.Parameter(torch.zeros(4))
        legacy_key = "place" + "holder"
        model.load_state_dict({legacy_key: torch.ones(4)}, strict=True)
        torch.testing.assert_close(model.feature_offset, torch.ones(4))


if __name__ == "__main__":
    unittest.main()
