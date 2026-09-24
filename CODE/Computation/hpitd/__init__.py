"""Hierarchical physics-integrated thermal-deformation prediction package."""

from .model import (
    HierarchicalPhysicsInformedTransformer,
    HierarchicalPhysicsIntegratedTransformer,
)

__all__ = [
    "HierarchicalPhysicsIntegratedTransformer",
    "HierarchicalPhysicsInformedTransformer",
]
__version__ = "0.1.0"
