from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ResizeCrop:
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
        raise ValueError("source and target sizes must be positive")
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
    raise ValueError(f"unsupported intrinsics shape {tuple(intrinsics.shape)}")


def intrinsics_vector_to_matrix(intrinsics: torch.Tensor) -> torch.Tensor:
    if intrinsics.shape[-1] != 4:
        raise ValueError(f"expected [...,4], got {tuple(intrinsics.shape)}")
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


def normalize_c2w_to_first_capture(
    c2w: torch.Tensor,
    first_capture_c2w: torch.Tensor,
) -> torch.Tensor:
    if c2w.shape[-2:] != (4, 4) or first_capture_c2w.shape[-2:] != (4, 4):
        raise ValueError("c2w tensors must end in [4,4]")
    first_w2c = torch.linalg.inv(first_capture_c2w)
    while first_w2c.ndim < c2w.ndim:
        first_w2c = first_w2c.unsqueeze(-3)
    return first_w2c @ c2w


def pixel_grid(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    v, u = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((u, v, torch.ones_like(u)), dim=-1)


def make_origin_direction_rays(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Equation (11) of GIM-World: world rays ``[origin, direction]``.

    Inputs may be ``[T,4,4]``/``[T,3,3]`` or batched equivalents.  Output is
    ``[..., H, W, 6]``.  ViPE and this implementation both use OpenCV +Z
    camera-forward convention.
    """

    if c2w.shape[:-2] != intrinsics.shape[:-2]:
        raise ValueError("pose and intrinsics leading dimensions must match")
    leading = c2w.shape[:-2]
    with torch.amp.autocast(c2w.device.type, enabled=False):
        flat_pose = c2w.reshape(-1, 4, 4).float()
        flat_k = intrinsics.reshape(-1, 3, 3).float()
        grid = pixel_grid(
            height,
            width,
            device=c2w.device,
            dtype=torch.float32,
        ).reshape(1, height * width, 3)
        grid = grid.expand(flat_pose.shape[0], -1, -1)
        directions_camera = torch.bmm(
            grid,
            torch.linalg.inv(flat_k).transpose(1, 2),
        )
        directions_world = torch.bmm(
            directions_camera,
            flat_pose[:, :3, :3].transpose(1, 2),
        )
        directions_world = F.normalize(directions_world, dim=-1, eps=1e-6)
        origins = flat_pose[:, None, :3, 3].expand_as(directions_world)
        rays = torch.cat((origins, directions_world), dim=-1)
    return rays.reshape(*leading, height, width, 6)


def camera_vector(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: Sequence[int],
) -> torch.Tensor:
    """Linear camera embedding input used by the LingBot adaptation.

    The paper only specifies a linear camera embedding and does not disclose
    its vectorization.  We use the 12 camera-to-world entries plus normalized
    ``fx, fy, cx, cy`` so variable source resolutions remain well-defined.
    """

    height, width = (float(value) for value in image_hw)
    pose = c2w[..., :3, :4].reshape(*c2w.shape[:-2], 12)
    calibration = torch.stack(
        (
            intrinsics[..., 0, 0] / width,
            intrinsics[..., 1, 1] / height,
            intrinsics[..., 0, 2] / width,
            intrinsics[..., 1, 2] / height,
        ),
        dim=-1,
    )
    return torch.cat((pose, calibration), dim=-1)
