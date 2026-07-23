from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sample_flow_sigmas(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    logits = torch.randn(
        batch_size,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    sigmas = torch.sigmoid(logit_mean + logit_std * logits)
    return sigmas.to(dtype)


def add_flow_noise(
    clean_latents: torch.Tensor,
    sigmas: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    known_prefix_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rectified-flow interpolation used by the LingBot base generator."""

    if noise is None:
        noise = torch.randn_like(clean_latents)
    sigma = sigmas.view(-1, *([1] * (clean_latents.ndim - 1)))
    noisy = (1.0 - sigma) * clean_latents + sigma * noise
    target_velocity = noise - clean_latents
    if known_prefix_mask is not None:
        prefix = known_prefix_mask.bool()
        noisy = torch.where(prefix, clean_latents, noisy)
    return noisy, target_velocity, noise


def masked_flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    known_prefix_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    error = (predicted_velocity.float() - target_velocity.float()).square()
    if known_prefix_mask is None:
        return error.mean()
    unknown = (~known_prefix_mask.bool()).to(error.dtype).expand_as(error)
    return (error * unknown).sum() / unknown.sum().clamp_min(1.0)


def predict_clean_latents(
    noisy_latents: torch.Tensor,
    predicted_velocity: torch.Tensor,
    sigmas: torch.Tensor,
    known_prefix_mask: Optional[torch.Tensor] = None,
    clean_prefix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sigma = sigmas.view(-1, *([1] * (noisy_latents.ndim - 1)))
    prediction = noisy_latents.float() - sigma.float() * predicted_velocity.float()
    if known_prefix_mask is not None and clean_prefix is not None:
        prediction = torch.where(
            known_prefix_mask.bool(), clean_prefix.float(), prediction
        )
    return prediction


def gather_flat_queries(
    values: torch.Tensor,
    query_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather ``(T,H,W)`` query locations from scalar or feature grids."""

    batch = values.shape[0]
    if values.ndim == 4:
        flat = values.reshape(batch, -1)
        return torch.gather(flat, 1, query_indices)
    if values.ndim == 5:
        flat = values.reshape(batch, -1, values.shape[-1])
        index = query_indices[..., None].expand(
            batch, query_indices.shape[1], values.shape[-1]
        )
        return torch.gather(flat, 1, index)
    raise ValueError("values must have shape (B,T,H,W) or (B,T,H,W,D)")


def scale_invariant_log_depth_loss(
    predicted_log_depth: torch.Tensor,
    target_depth: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    scale_weight: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    target_log_depth = target_depth.float().clamp_min(eps).log()
    residual = predicted_log_depth.float() - target_log_depth
    valid_depth = target_depth.float() > eps
    if confidence is None:
        confidence = valid_depth.to(residual.dtype)
    else:
        confidence = confidence.float().clamp_min(0.0) * valid_depth
    mean_residual = (residual * confidence).sum(dim=-1, keepdim=True) / confidence.sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    centered = residual - mean_residual
    denominator = confidence.sum().clamp_min(1.0)
    invariant = (centered.square() * confidence).sum() / denominator
    metric_scale = (residual.square() * confidence).sum() / denominator
    return invariant + scale_weight * metric_scale


def geometry_feature_cosine_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    confidence: Optional[torch.Tensor] = None,
    scale_weight: float = 0.1,
) -> torch.Tensor:
    angular_error = 1.0 - F.cosine_similarity(
        predicted.float(), target.float(), dim=-1
    )
    scale_error = F.smooth_l1_loss(
        predicted.float(), target.float(), reduction="none"
    ).mean(dim=-1)
    error = angular_error + scale_weight * scale_error
    if confidence is None:
        return error.mean()
    weight = confidence.float().clamp_min(0.0)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def inverse_warp_features(
    source_features: torch.Tensor,
    target_depth: torch.Tensor,
    source_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
    source_intrinsics: torch.Tensor,
    target_intrinsics: torch.Tensor,
    depth_is_ray_distance: bool = False,
    source_depth: Optional[torch.Tensor] = None,
    occlusion_relative_threshold: float = 0.05,
    occlusion_absolute_threshold: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp source features into a target view using target-view depth.

    Intrinsics are normalized by width and height. Camera translation and depth
    must use the same scale. This function is a training loss only; its output
    is never passed to the generator.
    """

    batch, _, height, width = source_features.shape
    if target_depth.shape[-2:] != (height, width):
        target_depth = F.interpolate(
            target_depth[:, None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    dtype = target_depth.dtype
    device = target_depth.device
    u = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    v = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu = uu[None].expand(batch, -1, -1)
    vv = vv[None].expand(batch, -1, -1)

    fx = target_intrinsics[:, 0, 0, None, None].clamp_min(1e-6)
    fy = target_intrinsics[:, 1, 1, None, None].clamp_min(1e-6)
    cx = target_intrinsics[:, 0, 2, None, None]
    cy = target_intrinsics[:, 1, 2, None, None]
    rays = torch.stack(
        ((uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)),
        dim=-1,
    )
    if depth_is_ray_distance:
        rays = F.normalize(rays, dim=-1, eps=1e-6)
    points_target = rays * target_depth[..., None]

    target_rotation = target_c2w[:, :3, :3]
    target_translation = target_c2w[:, :3, 3]
    points_world = torch.einsum(
        "bij,bhwj->bhwi", target_rotation, points_target
    ) + target_translation[:, None, None]
    source_rotation = source_c2w[:, :3, :3]
    source_translation = source_c2w[:, :3, 3]
    points_source = torch.einsum(
        "bij,bhwj->bhwi",
        source_rotation.transpose(-1, -2),
        points_world - source_translation[:, None, None],
    )

    source_z = points_source[..., 2]
    source_u = (
        source_intrinsics[:, 0, 0, None, None]
        * points_source[..., 0]
        / source_z.clamp_min(1e-6)
        + source_intrinsics[:, 0, 2, None, None]
    )
    source_v = (
        source_intrinsics[:, 1, 1, None, None]
        * points_source[..., 1]
        / source_z.clamp_min(1e-6)
        + source_intrinsics[:, 1, 2, None, None]
    )
    sample_grid = torch.stack((2.0 * source_u - 1.0, 2.0 * source_v - 1.0), dim=-1)
    warped = F.grid_sample(
        source_features.float(),
        sample_grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    valid = (
        (target_depth > 0)
        & (source_z > 1e-5)
        & (source_u >= 0.0)
        & (source_u <= 1.0)
        & (source_v >= 0.0)
        & (source_v <= 1.0)
    )
    if source_depth is not None:
        if source_depth.shape[-2:] != (height, width):
            source_depth = F.interpolate(
                source_depth[:, None].float(),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
        observed_source_depth = F.grid_sample(
            source_depth[:, None].float(),
            sample_grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )[:, 0]
        projected_source_depth = (
            torch.linalg.vector_norm(points_source.float(), dim=-1)
            if depth_is_ray_distance
            else source_z.float()
        )
        occlusion_tolerance = (
            occlusion_absolute_threshold
            + occlusion_relative_threshold
            * torch.maximum(
                observed_source_depth.abs(), projected_source_depth.abs()
            )
        )
        depth_agrees = (
            (observed_source_depth > 0)
            & (
                (observed_source_depth - projected_source_depth).abs()
                <= occlusion_tolerance
            )
        )
        valid = valid & depth_agrees
    return warped, valid


def multi_view_reprojection_loss(
    predicted_clean_latents: torch.Tensor,
    depth: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    pairs: torch.Tensor,
    pair_valid_mask: Optional[torch.Tensor] = None,
    depth_confidence: Optional[torch.Tensor] = None,
    depth_is_ray_distance: bool = False,
    occlusion_relative_threshold: float = 0.05,
    occlusion_absolute_threshold: float = 0.01,
) -> torch.Tensor:
    """Enforce physical-point consistency across loop closures or trajectories."""

    batch, _, frames, height, width = predicted_clean_latents.shape
    if c2w.shape[:2] != (batch, frames):
        raise ValueError("c2w must align with predicted latent frames")
    if pairs.ndim != 3 or pairs.shape[0] != batch or pairs.shape[-1] != 2:
        raise ValueError("pairs must have shape (B, P, 2)")
    if pair_valid_mask is None:
        pair_valid_mask = torch.ones(
            pairs.shape[:2], dtype=torch.bool, device=pairs.device
        )
    if pair_valid_mask.shape != pairs.shape[:2]:
        raise ValueError("pair_valid_mask must have shape (B, P)")
    active_indices = pairs[pair_valid_mask.bool()]
    if active_indices.numel() and (
        (active_indices < 0).any() or (active_indices >= frames).any()
    ):
        raise ValueError("Active consistency-pair indices are out of range")
    total_error = predicted_clean_latents.new_zeros((), dtype=torch.float32)
    total_weight = predicted_clean_latents.new_zeros((), dtype=torch.float32)

    for pair_index in range(pairs.shape[1]):
        active = pair_valid_mask[:, pair_index]
        if not active.any():
            continue
        batch_indices = torch.arange(batch, device=pairs.device)
        source_index = torch.where(
            active, pairs[:, pair_index, 0], torch.zeros_like(pairs[:, pair_index, 0])
        )
        target_index = torch.where(
            active, pairs[:, pair_index, 1], torch.zeros_like(pairs[:, pair_index, 1])
        )
        source = predicted_clean_latents[batch_indices, :, source_index]
        target = predicted_clean_latents[batch_indices, :, target_index]
        source_depth = depth[batch_indices, source_index]
        target_depth = depth[batch_indices, target_index]
        warped, valid = inverse_warp_features(
            source,
            target_depth,
            c2w[batch_indices, source_index],
            c2w[batch_indices, target_index],
            intrinsics[batch_indices, source_index],
            intrinsics[batch_indices, target_index],
            depth_is_ray_distance=depth_is_ray_distance,
            source_depth=source_depth,
            occlusion_relative_threshold=occlusion_relative_threshold,
            occlusion_absolute_threshold=occlusion_absolute_threshold,
        )
        weight = valid[:, None].float() * active[:, None, None, None].float()
        if depth_confidence is not None:
            confidence = depth_confidence[batch_indices, target_index]
            if confidence.shape[-2:] != (height, width):
                confidence = F.interpolate(
                    confidence[:, None].float(),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
            weight = weight * confidence[:, None].clamp_min(0.0)
        error = F.smooth_l1_loss(
            warped, target.float(), reduction="none", beta=0.05
        )
        total_error = total_error + (error * weight).sum()
        total_weight = total_weight + weight.sum() * target.shape[1]
    return total_error / total_weight.clamp_min(1.0)
