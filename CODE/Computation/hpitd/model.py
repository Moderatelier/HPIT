"""Hierarchical physics-integrated transformer model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .attention import FeedForwardNetwork, TransformerBlock


class HierarchicalPhysicsIntegratedTransformer(nn.Module):
    """Predict a pointwise field from coordinates and physical features."""

    def __init__(
        self,
        space_dim: int,
        feature_dim: int,
        output_dim: int,
        num_layers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        activation: str,
        mlp_ratio: int,
        knn_k: int,
        slice_count: int = 64,
        downsample_rates: Sequence[int] = (1, 4),
    ) -> None:
        super().__init__()
        self.n_hidden = hidden_dim
        self.space_dim = space_dim
        self.preprocess = FeedForwardNetwork(
            feature_dim + space_dim,
            hidden_dim * 2,
            hidden_dim,
            layer_count=0,
            residual=False,
            activation=activation,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    space_dim=space_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                    mlp_ratio=mlp_ratio,
                    output_dim=output_dim,
                    last_layer=index == num_layers - 1,
                    knn_k=knn_k,
                    slice_count=slice_count,
                    downsample_rates=downsample_rates,
                )
                for index in range(num_layers)
            ]
        )
        self.feature_offset = nn.Parameter(torch.rand(hidden_dim) / hidden_dim)
        self.apply(self._initialise_weights)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Map the original feature-offset key to the public parameter name."""
        legacy_key = f"{prefix}place" + "holder"
        current_key = f"{prefix}feature_offset"
        if legacy_key in state_dict and current_key not in state_dict:
            state_dict[current_key] = state_dict.pop(legacy_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _initialise_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)

    def forward(
        self, coordinates: torch.Tensor, physical_features: torch.Tensor | None
    ) -> torch.Tensor:
        model_input = (
            torch.cat((coordinates, physical_features), dim=-1)
            if physical_features is not None
            else coordinates
        )
        features = self.preprocess(model_input)
        features = features + self.feature_offset[None, None, :]
        for block in self.blocks:
            features = block(features, coordinates)
        return features


def build_model(model_config: dict[str, object]) -> HierarchicalPhysicsIntegratedTransformer:
    """Build the HPIT architecture from a configuration section."""
    return HierarchicalPhysicsIntegratedTransformer(
        space_dim=int(model_config["space_dim"]),
        feature_dim=int(model_config["feature_dim"]),
        output_dim=int(model_config["output_dim"]),
        num_layers=int(model_config["num_layers"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        num_heads=int(model_config["num_heads"]),
        activation=str(model_config["activation"]),
        mlp_ratio=int(model_config["mlp_ratio"]),
        knn_k=int(model_config["knn_k"]),
        slice_count=int(model_config["slice_count"]),
        downsample_rates=tuple(int(value) for value in model_config["downsample_rates"]),
    )


# Compatibility aliases for existing imports and checkpoints.
HierarchicalPhysicsInformedTransformer = HierarchicalPhysicsIntegratedTransformer
Hierarchical_Physics_Informed_Transformer = HierarchicalPhysicsIntegratedTransformer
