"""Hierarchical physics-integrated cross-attention layers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from .cra_pcn_backend import create_cross_relation_transformer, hierarchical_fps
from .point_ops import three_interpolate, three_nn


class PhysicsIntegratedAttention(nn.Module):
    """Aggregate points into slices, model cross-scale relations, and reconstruct points."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        knn_k: int,
        space_dim: int,
        slice_count: int,
        downsample_rates: Sequence[int],
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.space_dim = space_dim
        self.slice_num = slice_count

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.temperature = nn.Parameter(torch.ones(1, heads, 1, 1) * 0.5)
        self.in_project_slice = nn.Linear(dim_head, slice_count)
        nn.init.orthogonal_(self.in_project_slice.weight)

        neighbour_counts = (knn_k, max(1, knn_k // 2))
        self.crt_slice = create_cross_relation_transformer(
            dim_in=dim_head,
            is_inter=False,
            down_rates=downsample_rates,
            knns=neighbour_counts,
            space_dim=space_dim,
        )
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch_size, point_count, _ = features.shape
        positions_channels_first = positions.transpose(1, 2).contiguous()

        physics_features = self.in_project_fx(features).reshape(
            batch_size, point_count, self.heads, self.dim_head
        ).permute(0, 2, 1, 3).contiguous()
        spatial_features = self.in_project_x(features).reshape(
            batch_size, point_count, self.heads, self.dim_head
        ).permute(0, 2, 1, 3).contiguous()

        slice_weights = self.softmax(self.in_project_slice(spatial_features) / self.temperature)
        slice_norm = slice_weights.sum(dim=2)
        normaliser = (slice_norm + 1e-5).unsqueeze(-1)
        slice_physics = torch.einsum(
            "bhnc,bhng->bhgc", physics_features, slice_weights
        ) / normaliser
        slice_spatial = torch.einsum(
            "bhnc,bhng->bhgc", spatial_features, slice_weights
        ) / normaliser

        expanded_positions = positions_channels_first.unsqueeze(1).expand(
            -1, self.heads, -1, -1
        ).permute(0, 1, 3, 2).contiguous()
        slice_positions = torch.einsum(
            "bhng,bhnc->bhgc", slice_weights, expanded_positions
        ) / normaliser
        slice_positions = slice_positions.permute(0, 1, 3, 2).contiguous()

        merged_positions = slice_positions.reshape(
            batch_size * self.heads, self.space_dim, self.slice_num
        ).contiguous()
        fps_indices = hierarchical_fps(merged_positions, self.crt_slice.down_rates)
        merged_physics = slice_physics.reshape(
            batch_size * self.heads, self.slice_num, self.dim_head
        ).permute(0, 2, 1).contiguous()
        merged_spatial = slice_spatial.reshape(
            batch_size * self.heads, self.slice_num, self.dim_head
        ).permute(0, 2, 1).contiguous()

        aggregate, _, _, reduced_positions = self.crt_slice(
            [merged_positions, merged_physics],
            [merged_positions, merged_spatial],
            fps_idxs_q=fps_indices,
            fps_idxs_s=fps_indices,
            num_groups=self.heads,
        )
        reduced_count = aggregate.shape[-1]
        cross_scale_features = aggregate.permute(0, 2, 1).reshape(
            batch_size, self.heads, reduced_count, self.dim_head
        ).contiguous()
        reduced_positions = reduced_positions.reshape(
            batch_size, self.heads, self.space_dim, reduced_count
        ).contiguous()

        attended = scaled_dot_product_attention(
            self.to_q(cross_scale_features),
            self.to_k(cross_scale_features),
            self.to_v(cross_scale_features),
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False,
        ).contiguous()

        attended_merged = attended.reshape(
            batch_size * self.heads, reduced_count, self.dim_head
        ).permute(0, 2, 1).contiguous()
        reduced_positions_merged = reduced_positions.reshape(
            batch_size * self.heads, self.space_dim, reduced_count
        ).contiguous()
        full_slice_positions = slice_positions.reshape(
            batch_size * self.heads, self.space_dim, self.slice_num
        ).contiguous()
        distances, indices = three_nn(
            full_slice_positions.transpose(1, 2).contiguous(),
            reduced_positions_merged.transpose(1, 2).contiguous(),
        )
        weights = torch.clamp(1.0 / (distances + 1e-8), max=1e5)
        weights = weights / weights.sum(dim=2, keepdim=True)
        full_slice_features = three_interpolate(
            attended_merged, indices.contiguous(), weights.contiguous()
        ).contiguous()

        full_slice_features = full_slice_features.permute(0, 2, 1).reshape(
            batch_size, self.heads, self.slice_num, self.dim_head
        ).contiguous()
        reconstructed = torch.einsum(
            "bhgc,bhng->bhnc", full_slice_features, slice_weights
        )
        reconstructed = reconstructed.permute(0, 2, 1, 3).reshape(
            batch_size, point_count, self.heads * self.dim_head
        ).contiguous()
        return self.to_out(reconstructed).contiguous()


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "silu": nn.SiLU,
}


class FeedForwardNetwork(nn.Module):
    """Configurable fully connected residual network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        layer_count: int = 1,
        activation: str = "gelu",
        residual: bool = True,
    ) -> None:
        super().__init__()
        activation_key = activation.lower()
        if activation_key not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        activation_type = ACTIVATIONS[activation_key]
        self.layer_count = layer_count
        self.residual = residual
        self.linear_pre = nn.Sequential(nn.Linear(input_dim, hidden_dim), activation_type())
        self.linear_post = nn.Linear(hidden_dim, output_dim)
        self.linears = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), activation_type())
                for _ in range(layer_count)
            ]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.linear_pre(tensor)
        for layer in self.linears:
            tensor = layer(tensor) + tensor if self.residual else layer(tensor)
        return self.linear_post(tensor)


class TransformerBlock(nn.Module):
    """One residual attention and feed-forward block."""

    def __init__(
        self,
        space_dim: int,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        activation: str = "gelu",
        mlp_ratio: int = 4,
        last_layer: bool = False,
        output_dim: int = 2,
        knn_k: int = 16,
        slice_count: int = 64,
        downsample_rates: Sequence[int] = (1, 4),
    ) -> None:
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = PhysicsIntegratedAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            knn_k=knn_k,
            space_dim=space_dim,
            slice_count=slice_count,
            downsample_rates=downsample_rates,
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = FeedForwardNetwork(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            layer_count=0,
            residual=False,
            activation=activation,
        )
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, features: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        features = self.Attn(self.ln_1(features), positions) + features
        features = self.mlp(self.ln_2(features)) + features
        if self.last_layer:
            return self.mlp2(self.ln_3(features))
        return features


# Compatibility aliases for existing imports and checkpoints.
PhysicsInformedAttention = PhysicsIntegratedAttention
Physics_Informed_Attention = PhysicsIntegratedAttention
MLP = FeedForwardNetwork
Transformer_block = TransformerBlock
