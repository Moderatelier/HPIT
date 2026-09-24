"""Point-cloud operations with an optional CUDA extension backend."""

from __future__ import annotations

import torch

try:
    from pointnet2_ops.pointnet2_utils import (
        ball_query as _cuda_ball_query,
        furthest_point_sample as _cuda_furthest_point_sample,
        gather_operation as _cuda_gather_operation,
        grouping_operation as _cuda_grouping_operation,
        three_interpolate as _cuda_three_interpolate,
        three_nn as _cuda_three_nn,
    )

    POINTNET2_AVAILABLE = True
except (ImportError, OSError):
    POINTNET2_AVAILABLE = False


def backend_name(tensor: torch.Tensor | None = None) -> str:
    """Report the backend selected for a tensor."""
    if POINTNET2_AVAILABLE and tensor is not None and tensor.is_cuda:
        return "pointnet2_ops"
    return "pytorch"


def furthest_point_sample(xyz: torch.Tensor, sample_count: int) -> torch.Tensor:
    """Return farthest-point indices for `xyz` shaped `(B, N, C)`."""
    if POINTNET2_AVAILABLE and xyz.is_cuda:
        return _cuda_furthest_point_sample(xyz.contiguous(), sample_count)
    if xyz.ndim != 3:
        raise ValueError("xyz must have shape (batch, points, coordinates)")
    batch_size, point_count, _ = xyz.shape
    if not 1 <= sample_count <= point_count:
        raise ValueError("sample_count must be between one and the point count")

    centroids = torch.zeros(batch_size, sample_count, dtype=torch.long, device=xyz.device)
    minimum_distance = torch.full(
        (batch_size, point_count), float("inf"), dtype=xyz.dtype, device=xyz.device
    )
    farthest = torch.zeros(batch_size, dtype=torch.long, device=xyz.device)
    batch_indices = torch.arange(batch_size, device=xyz.device)
    for index in range(sample_count):
        centroids[:, index] = farthest
        centroid = xyz[batch_indices, farthest].unsqueeze(1)
        distance = torch.sum((xyz - centroid) ** 2, dim=-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        farthest = torch.max(minimum_distance, dim=-1).indices
    return centroids.to(torch.int32)


def gather_operation(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather `(B, C, N)` features with `(B, S)` indices."""
    if POINTNET2_AVAILABLE and features.is_cuda:
        return _cuda_gather_operation(features.contiguous(), indices.to(torch.int32).contiguous())
    index = indices.to(torch.long).unsqueeze(1).expand(-1, features.shape[1], -1)
    return torch.gather(features, dim=2, index=index)


def grouping_operation(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather neighbourhood features using `(B, S, K)` indices."""
    if POINTNET2_AVAILABLE and features.is_cuda:
        return _cuda_grouping_operation(features.contiguous(), indices.to(torch.int32).contiguous())
    batch_size, channel_count, _ = features.shape
    sample_count, neighbour_count = indices.shape[1:]
    expanded = features.unsqueeze(2).expand(-1, -1, sample_count, -1)
    index = indices.to(torch.long).unsqueeze(1).expand(
        batch_size, channel_count, sample_count, neighbour_count
    )
    return torch.gather(expanded, dim=3, index=index)


def ball_query(
    radius: float,
    neighbour_count: int,
    xyz: torch.Tensor,
    query_xyz: torch.Tensor,
) -> torch.Tensor:
    """Return indices inside a radius, padding with the nearest valid point."""
    if POINTNET2_AVAILABLE and xyz.is_cuda:
        return _cuda_ball_query(radius, neighbour_count, xyz.contiguous(), query_xyz.contiguous())
    distances = torch.cdist(query_xyz, xyz)
    sorted_distances, sorted_indices = torch.sort(distances, dim=-1)
    selected_indices = sorted_indices[..., :neighbour_count]
    selected_distances = sorted_distances[..., :neighbour_count]
    first_index = selected_indices[..., :1]
    selected_indices = torch.where(
        selected_distances <= radius,
        selected_indices,
        first_index.expand_as(selected_indices),
    )
    if selected_indices.shape[-1] < neighbour_count:
        selected_indices = torch.cat(
            [
                selected_indices,
                first_index.expand(
                    *selected_indices.shape[:-1], neighbour_count - selected_indices.shape[-1]
                ),
            ],
            dim=-1,
        )
    return selected_indices.to(torch.int32)


def three_nn(unknown: torch.Tensor, known: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the three nearest known points for every unknown point."""
    if POINTNET2_AVAILABLE and unknown.is_cuda:
        return _cuda_three_nn(unknown.contiguous(), known.contiguous())
    distance = torch.cdist(unknown, known)
    neighbour_count = min(3, known.shape[1])
    values, indices = torch.topk(distance, k=neighbour_count, dim=-1, largest=False, sorted=True)
    if neighbour_count < 3:
        pad_count = 3 - neighbour_count
        values = torch.cat([values, values[..., -1:].expand(*values.shape[:-1], pad_count)], dim=-1)
        indices = torch.cat(
            [indices, indices[..., -1:].expand(*indices.shape[:-1], pad_count)], dim=-1
        )
    return values, indices.to(torch.int32)


def three_interpolate(
    features: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Interpolate `(B, C, M)` features at `(B, N, 3)` neighbour indices."""
    if POINTNET2_AVAILABLE and features.is_cuda:
        return _cuda_three_interpolate(
            features.contiguous(), indices.to(torch.int32).contiguous(), weights.contiguous()
        )
    grouped = grouping_operation(features, indices)
    return torch.sum(grouped * weights.unsqueeze(1), dim=-1)
