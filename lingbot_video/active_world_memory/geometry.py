from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_c2w_to_first_capture(
    c2w: torch.Tensor,
    origin_c2w: torch.Tensor,
) -> torch.Tensor:
    """Express every camera-to-world matrix in the first-capture coordinate frame."""

    if c2w.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"expected (...,3,4) or (...,4,4) c2w, got {tuple(c2w.shape)}")
    if origin_c2w.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError("origin_c2w must be a 3x4 or 4x4 matrix")

    def homogeneous(value: torch.Tensor) -> torch.Tensor:
        if value.shape[-2:] == (4, 4):
            return value
        bottom = torch.zeros(
            *value.shape[:-2], 1, 4, dtype=value.dtype, device=value.device
        )
        bottom[..., 0, 3] = 1
        return torch.cat((value, bottom), dim=-2)

    c2w_h = homogeneous(c2w)
    origin_h = homogeneous(origin_c2w)
    normalized = torch.linalg.inv(origin_h) @ c2w_h
    return normalized


def resize_crop_intrinsics(
    intrinsics: torch.Tensor,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
) -> torch.Tensor:
    """Update K for resize-to-cover followed by a centered crop."""

    source_h, source_w = source_hw
    target_h, target_w = target_hw
    scale = max(target_h / source_h, target_w / source_w)
    resized_h = source_h * scale
    resized_w = source_w * scale
    crop_y = (resized_h - target_h) / 2.0
    crop_x = (resized_w - target_w) / 2.0

    output = intrinsics.clone().float()
    output[..., 0, 0] *= scale
    output[..., 1, 1] *= scale
    output[..., 0, 2] = output[..., 0, 2] * scale - crop_x
    output[..., 1, 2] = output[..., 1, 2] * scale - crop_y
    return output


def intrinsics_vector_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Convert ViPE intrinsics in either matrix or fx,fy,cx,cy form to K."""

    value = value.float()
    if value.shape[-2:] == (3, 3):
        return value
    flat = value.reshape(*value.shape[:-1], -1)
    if flat.shape[-1] == 9:
        return flat.reshape(*flat.shape[:-1], 3, 3)
    if flat.shape[-1] < 4:
        raise ValueError(f"unsupported intrinsics shape {tuple(value.shape)}")
    fx, fy, cx, cy = flat.unbind(dim=-1)[:4]
    output = torch.zeros(*flat.shape[:-1], 3, 3, dtype=flat.dtype, device=flat.device)
    output[..., 0, 0] = fx
    output[..., 1, 1] = fy
    output[..., 0, 2] = cx
    output[..., 1, 2] = cy
    output[..., 2, 2] = 1
    return output


def camera_descriptor(
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw: tuple[int, int],
    times: torch.Tensor | None = None,
) -> torch.Tensor:
    """A scale-stable camera descriptor without rendering any 3D proxy.

    Rotation uses the first two camera axes (continuous 6D representation),
    followed by translation, normalized intrinsics and normalized time.
    """

    c2w = c2w.float()
    intrinsics = intrinsics.float()
    rotation6d = c2w[..., :3, :2].reshape(*c2w.shape[:-2], 6)
    translation = c2w[..., :3, 3]
    height, width = image_hw
    intrinsics4 = torch.stack(
        (
            intrinsics[..., 0, 0] / width,
            intrinsics[..., 1, 1] / height,
            intrinsics[..., 0, 2] / width,
            intrinsics[..., 1, 2] / height,
        ),
        dim=-1,
    )
    if times is None:
        time_feature = torch.zeros_like(translation[..., :1])
    else:
        # Candidate memory and target chunks are encoded in separate calls, so
        # sequence-dependent normalization would map the same scene timestamp
        # to two different values.  Internal room-tour indices already use a
        # stable 0,5,10,... source-frame grid; 1024 covers a typical 5000-frame
        # source clip while remaining well behaved for longer trajectories.
        time_feature = (times.float() / 1024.0).unsqueeze(-1)
    return torch.cat((rotation6d, translation, intrinsics4, time_feature), dim=-1)


def pairwise_camera_relevance(
    target_c2w: torch.Tensor,
    memory_c2w: torch.Tensor,
    *,
    position_temperature: float = 1.0,
    direction_temperature: float = 0.5,
) -> torch.Tensor:
    """Frustum-proximity teacher used only to bootstrap the retriever.

    This is not a point-cloud projection and is never fed to the generator.  It
    returns a soft target with shape ``(B, target_views, memory_views)``.
    """

    target_position = target_c2w[..., :3, 3]
    memory_position = memory_c2w[..., :3, 3]
    distance = torch.cdist(target_position.float(), memory_position.float())

    target_forward = F.normalize(target_c2w[..., :3, 2].float(), dim=-1)
    memory_forward = F.normalize(memory_c2w[..., :3, 2].float(), dim=-1)
    direction = torch.einsum("btd,bmd->btm", target_forward, memory_forward)
    direction = ((direction + 1.0) / 2.0).clamp(0.0, 1.0)
    return torch.exp(-distance / max(position_temperature, 1e-6)) * torch.exp(
        -(1.0 - direction) / max(direction_temperature, 1e-6)
    )
