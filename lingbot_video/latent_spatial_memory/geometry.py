from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ResizeCrop:
    """Aspect-preserving resize followed by a centered crop."""

    scale_x: float
    scale_y: float
    crop_left: float
    crop_top: float
    resized_height: int
    resized_width: int
    target_height: int
    target_width: int


def center_resize_crop(
    source_hw: Sequence[int],
    target_hw: Sequence[int],
) -> ResizeCrop:
    source_h, source_w = (int(value) for value in source_hw)
    target_h, target_w = (int(value) for value in target_hw)
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("source and target image sizes must be positive")
    scale = max(target_w / source_w, target_h / source_h)
    resized_w = max(target_w, int(round(source_w * scale)))
    resized_h = max(target_h, int(round(source_h * scale)))
    crop_left = float(int(round((resized_w - target_w) / 2.0)))
    crop_top = float(int(round((resized_h - target_h) / 2.0)))
    return ResizeCrop(
        scale_x=resized_w / source_w,
        scale_y=resized_h / source_h,
        crop_left=crop_left,
        crop_top=crop_top,
        resized_height=resized_h,
        resized_width=resized_w,
        target_height=target_h,
        target_width=target_w,
    )


def resize_crop_intrinsics(
    intrinsics: torch.Tensor,
    source_hw: Sequence[int],
    target_hw: Sequence[int],
) -> torch.Tensor:
    """Transform pixel intrinsics for the exact resize-and-center-crop operation."""

    transform = center_resize_crop(source_hw, target_hw)
    if intrinsics.shape[-2:] == (3, 3):
        output = intrinsics.clone()
        output[..., 0, 0] *= transform.scale_x
        output[..., 1, 1] *= transform.scale_y
        output[..., 0, 2] = (
            (output[..., 0, 2] + 0.5) * transform.scale_x
            - 0.5
            - transform.crop_left
        )
        output[..., 1, 2] = (
            (output[..., 1, 2] + 0.5) * transform.scale_y
            - 0.5
            - transform.crop_top
        )
        return output
    if intrinsics.shape[-1] == 4:
        output = intrinsics.clone()
        output[..., 0] *= transform.scale_x
        output[..., 1] *= transform.scale_y
        output[..., 2] = (
            (output[..., 2] + 0.5) * transform.scale_x
            - 0.5
            - transform.crop_left
        )
        output[..., 3] = (
            (output[..., 3] + 0.5) * transform.scale_y
            - 0.5
            - transform.crop_top
        )
        return output
    raise ValueError(
        "intrinsics must end in [3,3] or [fx,fy,cx,cy], "
        f"got {tuple(intrinsics.shape)}"
    )


def intrinsics_vector_to_matrix(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.shape[-1] != 4:
        raise ValueError(f"expected [...,4] intrinsics, got {tuple(intrinsics.shape)}")
    output = torch.zeros(
        *intrinsics.shape[:-1],
        3,
        3,
        dtype=intrinsics.dtype,
        device=intrinsics.device,
    )
    output[..., 0, 0] = intrinsics[..., 0]
    output[..., 1, 1] = intrinsics[..., 1]
    output[..., 0, 2] = intrinsics[..., 2]
    output[..., 1, 2] = intrinsics[..., 3]
    output[..., 2, 2] = 1
    return output


def scale_intrinsics(
    intrinsics: torch.Tensor,
    source_hw: Sequence[int],
    target_hw: Sequence[int],
) -> torch.Tensor:
    """Scale a 3x3 pinhole matrix between grids with the same field of view."""

    source_h, source_w = (int(value) for value in source_hw)
    target_h, target_w = (int(value) for value in target_hw)
    if intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"expected [...,3,3] intrinsics, got {tuple(intrinsics.shape)}")
    output = intrinsics.clone()
    scale_x = target_w / source_w
    scale_y = target_h / source_h
    output[..., 0, 0] *= scale_x
    output[..., 1, 1] *= scale_y
    output[..., 0, 2] = (output[..., 0, 2] + 0.5) * scale_x - 0.5
    output[..., 1, 2] = (output[..., 1, 2] + 0.5) * scale_y - 0.5
    return output


def normalize_c2w_to_first_capture(
    c2w: torch.Tensor,
    first_capture_c2w: torch.Tensor,
) -> torch.Tensor:
    """Put the first capture camera at the origin with identity orientation.

    ViPE stores OpenCV camera-to-world matrices.  Left multiplication by the
    first camera's world-to-camera matrix changes the world coordinate system
    without changing any camera-relative geometry.
    """

    if c2w.shape[-2:] != (4, 4) or first_capture_c2w.shape[-2:] != (4, 4):
        raise ValueError("c2w inputs must end in [4,4]")
    first_w2c = torch.linalg.inv(first_capture_c2w)
    while first_w2c.ndim < c2w.ndim:
        first_w2c = first_w2c.unsqueeze(-3)
    return first_w2c @ c2w


def resize_crop_tensor(
    tensor: torch.Tensor,
    source_hw: Sequence[int],
    target_hw: Sequence[int],
    *,
    mode: str,
) -> torch.Tensor:
    """Apply the image-space crop used by the dataset to arbitrary channels."""

    transform = center_resize_crop(source_hw, target_hw)
    original_shape = tensor.shape
    if tensor.ndim < 2:
        raise ValueError("tensor must have spatial dimensions")
    flat = tensor.reshape(-1, 1, original_shape[-2], original_shape[-1])
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    resized = F.interpolate(
        flat.float(),
        size=(transform.resized_height, transform.resized_width),
        mode=mode,
        align_corners=align_corners,
    )
    left = int(round(transform.crop_left))
    top = int(round(transform.crop_top))
    cropped = resized[
        ...,
        top : top + transform.target_height,
        left : left + transform.target_width,
    ]
    return cropped.reshape(*original_shape[:-2], transform.target_height, transform.target_width)


def downsample_depth_bilinear(depth: torch.Tensor, target_hw: Sequence[int]) -> torch.Tensor:
    """Bilinear depth downsampling used by Mirage (Appendix B, Table 5)."""

    target_h, target_w = (int(value) for value in target_hw)
    original_shape = depth.shape
    flat = depth.reshape(-1, 1, original_shape[-2], original_shape[-1]).float()
    output = F.interpolate(flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return output.reshape(*original_shape[:-2], target_h, target_w)


def depth_validity_mask(
    depth: torch.Tensor,
    *,
    min_depth: float,
    max_depth: float,
    relative_edge_threshold: float,
) -> torch.Tensor:
    valid = torch.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if relative_edge_threshold <= 0:
        return valid
    flat = depth.reshape(-1, 1, depth.shape[-2], depth.shape[-1])
    safe_low = torch.where(
        torch.isfinite(flat),
        flat,
        torch.full_like(flat, float(max_depth)),
    )
    safe_high = torch.where(torch.isfinite(flat), flat, torch.zeros_like(flat))
    local_min = -F.max_pool2d(-safe_low, kernel_size=3, stride=1, padding=1)
    local_max = F.max_pool2d(safe_high, kernel_size=3, stride=1, padding=1)
    spread = (local_max - local_min) / flat.clamp_min(1e-6)
    edge_valid = torch.isfinite(spread) & (spread <= relative_edge_threshold)
    return valid & edge_valid.reshape_as(depth)


def pixel_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return homogeneous pixel-centre coordinates [u+0.5,v+0.5,1]."""

    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    return torch.stack((u, v, torch.ones_like(u)), dim=-1)


def backproject_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
) -> torch.Tensor:
    """Back-project OpenCV z-depth to world coordinates."""

    if depth.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {tuple(depth.shape)}")
    if intrinsics.shape != (3, 3) or c2w.shape != (4, 4):
        raise ValueError("intrinsics and c2w must be [3,3] and [4,4]")
    height, width = depth.shape
    grid = pixel_grid(height, width, device=depth.device, dtype=torch.float32)
    rays_camera = grid.reshape(-1, 3) @ torch.linalg.inv(intrinsics.float()).T
    points_camera = rays_camera * depth.float().reshape(-1, 1)
    points_world = points_camera @ c2w[:3, :3].float().T + c2w[:3, 3].float()
    return points_world.reshape(height, width, 3)


def make_plucker_rays(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Construct Pluecker rays [direction, origin x direction].

    Inputs may be [T,4,4]/[T,3,3] or [B,T,4,4]/[B,T,3,3].  The output is
    [...,6,H,W].  Poses should already be normalized to the first capture.
    """

    if c2w.shape[:-2] != intrinsics.shape[:-2]:
        raise ValueError("pose and intrinsics leading dimensions must match")
    leading = c2w.shape[:-2]
    flat_c2w = c2w.reshape(-1, 4, 4).float()
    flat_k = intrinsics.reshape(-1, 3, 3).float()
    grid = pixel_grid(height, width, device=c2w.device, dtype=torch.float32)
    grid = grid.reshape(1, height * width, 3).expand(flat_c2w.shape[0], -1, -1)
    directions_camera = torch.bmm(grid, torch.linalg.inv(flat_k).transpose(1, 2))
    directions_world = torch.bmm(directions_camera, flat_c2w[:, :3, :3].transpose(1, 2))
    directions_world = F.normalize(directions_world, dim=-1, eps=1e-6)
    origins = flat_c2w[:, None, :3, 3].expand_as(directions_world)
    moments = torch.cross(origins, directions_world, dim=-1)
    rays = torch.cat((directions_world, moments), dim=-1)
    rays = rays.reshape(*leading, height, width, 6)
    permutation = list(range(len(leading))) + [len(leading) + 2, len(leading), len(leading) + 1]
    return rays.permute(*permutation).contiguous()
