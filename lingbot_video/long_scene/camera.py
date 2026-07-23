from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_matrix_to_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Return the first two rotation columns in a continuous 6D representation."""

    return rotation[..., :, :2].transpose(-1, -2).reshape(*rotation.shape[:-2], 6)


def normalize_pixel_intrinsics(
    intrinsics: torch.Tensor,
    image_hw: torch.Tensor,
) -> torch.Tensor:
    """Convert pixel intrinsics to width/height-normalized intrinsics.

    Args:
        intrinsics: ``(..., 3, 3)`` pixel-space camera matrices.
        image_hw: ``(..., 2)`` values in ``(height, width)`` order.
    """

    height, width = image_hw.unbind(dim=-1)
    output = intrinsics.clone()
    output[..., 0, 0] = output[..., 0, 0] / width
    output[..., 0, 2] = output[..., 0, 2] / width
    output[..., 1, 1] = output[..., 1, 1] / height
    output[..., 1, 2] = output[..., 1, 2] / height
    return output


def compute_scene_normalization(
    source_c2w: torch.Tensor,
    source_valid_mask: Optional[torch.Tensor] = None,
    quantile: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robustly normalize arbitrary SLAM world coordinates per scene."""

    if source_c2w.ndim != 4 or source_c2w.shape[-2:] != (4, 4):
        raise ValueError("source_c2w must have shape (B, N, 4, 4)")
    batch, views = source_c2w.shape[:2]
    if source_valid_mask is None:
        source_valid_mask = torch.ones(
            batch, views, dtype=torch.bool, device=source_c2w.device
        )
    else:
        source_valid_mask = source_valid_mask.bool()
    centers = source_c2w[..., :3, 3]
    scene_centers = []
    scene_scales = []
    for batch_index in range(batch):
        valid_centers = centers[batch_index, source_valid_mask[batch_index]]
        if valid_centers.numel() == 0:
            raise ValueError("Every scene must contain at least one valid source camera")
        center = valid_centers.median(dim=0).values
        radius = torch.linalg.vector_norm(valid_centers - center, dim=-1)
        scale = torch.quantile(radius.float(), quantile).to(radius.dtype).clamp_min(1e-3)
        scene_centers.append(center)
        scene_scales.append(scale)
    return torch.stack(scene_centers), torch.stack(scene_scales)


def compute_world_to_scene_rotation(
    source_c2w: torch.Tensor,
    source_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Use the first valid capture camera as a stable per-scene coordinate frame."""

    if source_c2w.ndim != 4 or source_c2w.shape[-2:] != (4, 4):
        raise ValueError("source_c2w must have shape (B, N, 4, 4)")
    batch, views = source_c2w.shape[:2]
    if source_valid_mask is None:
        source_valid_mask = torch.ones(
            batch, views, dtype=torch.bool, device=source_c2w.device
        )
    else:
        source_valid_mask = source_valid_mask.bool()
    if not source_valid_mask.any(dim=1).all():
        raise ValueError("Every scene must contain at least one valid source camera")
    first_valid = source_valid_mask.to(torch.int64).argmax(dim=1)
    batch_indices = torch.arange(batch, device=source_c2w.device)
    reference_rotation = source_c2w[
        batch_indices, first_valid, :3, :3
    ]
    return reference_rotation.transpose(-1, -2)


def normalize_camera_poses(
    c2w: torch.Tensor,
    scene_center: torch.Tensor,
    scene_scale: torch.Tensor,
    world_to_scene_rotation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform arbitrary SLAM coordinates into a canonical, unit-scale scene frame."""

    if c2w.shape[-2:] != (4, 4):
        raise ValueError("c2w must end in (4, 4)")
    scene_center = scene_center.to(device=c2w.device, dtype=c2w.dtype)
    scene_scale = scene_scale.to(device=c2w.device, dtype=c2w.dtype)
    expand_dims = c2w.ndim - 3
    center = scene_center.view(scene_center.shape[0], *([1] * expand_dims), 3)
    scale = scene_scale.view(scene_scale.shape[0], *([1] * expand_dims), 1)
    if world_to_scene_rotation is None:
        world_to_scene_rotation = torch.eye(
            3, dtype=c2w.dtype, device=c2w.device
        )[None].expand(c2w.shape[0], -1, -1)
    else:
        world_to_scene_rotation = world_to_scene_rotation.to(
            device=c2w.device, dtype=c2w.dtype
        )
    if world_to_scene_rotation.shape != (c2w.shape[0], 3, 3):
        raise ValueError("world_to_scene_rotation must have shape (B, 3, 3)")
    rotation = world_to_scene_rotation.view(
        c2w.shape[0], *([1] * expand_dims), 3, 3
    )
    output = c2w.clone()
    output[..., :3, :3] = rotation @ c2w[..., :3, :3]
    translated = (c2w[..., :3, 3] - center) / scale
    output[..., :3, 3] = (rotation @ translated.unsqueeze(-1)).squeeze(-1)
    return output


def view_pose_features(c2w: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
    """Compact per-view features injected into recurrent observation tokens."""

    if c2w.shape[:-2] != intrinsics.shape[:-2]:
        raise ValueError("c2w and intrinsics must share leading dimensions")
    rotation = rotation_matrix_to_6d(c2w[..., :3, :3])
    translation = c2w[..., :3, 3]
    calibration = torch.stack(
        (
            intrinsics[..., 0, 0],
            intrinsics[..., 1, 1],
            intrinsics[..., 0, 2],
            intrinsics[..., 1, 2],
        ),
        dim=-1,
    )
    return torch.cat((translation, rotation, calibration), dim=-1)


def relative_motion_features(c2w: torch.Tensor) -> torch.Tensor:
    """Encode frame-to-frame SE(3) motion without a matrix logarithm."""

    previous = torch.cat((c2w[:, :1], c2w[:, :-1]), dim=1)
    previous_rotation = previous[..., :3, :3]
    current_rotation = c2w[..., :3, :3]
    relative_rotation = previous_rotation.transpose(-1, -2) @ current_rotation
    translation_world = c2w[..., :3, 3] - previous[..., :3, 3]
    relative_translation = (
        previous_rotation.transpose(-1, -2) @ translation_world.unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat((relative_translation, rotation_matrix_to_6d(relative_rotation)), dim=-1)


def plucker_rays(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build world-space Plücker rays on a normalized image grid.

    ``intrinsics`` must be normalized by image width/height. The output is
    ``(B, T, H, W, 9)`` with camera origin, unit direction, and moment.
    """

    if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
        raise ValueError("c2w must have shape (B, T, 4, 4)")
    if intrinsics.shape != c2w.shape[:-2] + (3, 3):
        raise ValueError("intrinsics must have shape (B, T, 3, 3)")

    dtype = c2w.dtype
    device = c2w.device
    u = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    v = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    uu = uu.view(1, 1, height, width)
    vv = vv.view(1, 1, height, width)

    fx = intrinsics[..., 0, 0, None, None].clamp_min(1e-6)
    fy = intrinsics[..., 1, 1, None, None].clamp_min(1e-6)
    cx = intrinsics[..., 0, 2, None, None]
    cy = intrinsics[..., 1, 2, None, None]
    directions_camera = torch.stack(
        ((uu - cx) / fx, (vv - cy) / fy, torch.ones_like((uu - cx) / fx)),
        dim=-1,
    )

    rotation = c2w[..., :3, :3]
    directions_world = torch.einsum(
        "btij,bthwj->bthwi", rotation, directions_camera
    )
    directions_world = F.normalize(directions_world, dim=-1, eps=1e-6)
    origins = c2w[..., :3, 3].unsqueeze(-2).unsqueeze(-2).expand_as(directions_world)
    moments = torch.linalg.cross(origins, directions_world, dim=-1)
    return torch.cat((origins, directions_world, moments), dim=-1)


class RayFourierEncoder(nn.Module):
    """Fourier-encode Plücker rays and project them to a model width."""

    def __init__(self, output_dim: int, bands: int = 6, input_dim: int = 9):
        super().__init__()
        self.input_dim = input_dim
        self.bands = bands
        frequencies = (2.0 ** torch.arange(bands, dtype=torch.float32)) * math.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        encoded_dim = input_dim * (1 + 2 * bands)
        self.projection = nn.Sequential(
            nn.Linear(encoded_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, rays: torch.Tensor) -> torch.Tensor:
        if rays.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected ray dimension {self.input_dim}, got {rays.shape[-1]}"
            )
        phase = rays.unsqueeze(-1).float() * self.frequencies
        encoded = torch.cat(
            (rays.float(), phase.sin().flatten(-2), phase.cos().flatten(-2)),
            dim=-1,
        )
        return self.projection(encoded)


class CameraTokenEncoder(nn.Module):
    """Encode absolute world rays and local camera motion as visual-token bias."""

    def __init__(self, hidden_size: int, ray_fourier_bands: int = 6):
        super().__init__()
        self.ray_encoder = RayFourierEncoder(hidden_size, ray_fourier_bands)
        self.motion_encoder = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        grid_height: int,
        grid_width: int,
    ) -> torch.Tensor:
        rays = plucker_rays(c2w, intrinsics, grid_height, grid_width)
        ray_tokens = self.ray_encoder(rays)
        motion = self.motion_encoder(relative_motion_features(c2w).float())
        ray_tokens = ray_tokens + motion[:, :, None, None, :]
        batch = ray_tokens.shape[0]
        return self.output_gate.tanh() * ray_tokens.reshape(batch, -1, ray_tokens.shape[-1])
