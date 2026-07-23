from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset


REQUIRED_BATCH_KEYS = {
    "capture_latents",
    "capture_c2w",
    "capture_intrinsics",
    "capture_valid_mask",
    "target_latents",
    "target_c2w",
    "target_intrinsics",
    "target_depth",
    "prompt_embeds",
    "prompt_attention_mask",
}


def _validate_trajectory(
    batch: dict[str, torch.Tensor],
    *,
    prefix: str,
    batch_size: int,
    channels: int,
    height: int,
    width: int,
) -> None:
    latents = batch[f"{prefix}target_latents"]
    if (
        latents.ndim != 5
        or latents.shape[0] != batch_size
        or latents.shape[1] != channels
        or latents.shape[-2:] != (height, width)
    ):
        raise ValueError(
            f"{prefix}target_latents must have shape (B,C,T,H,W)"
        )
    frames = latents.shape[2]
    if batch[f"{prefix}target_c2w"].shape != (batch_size, frames, 4, 4):
        raise ValueError(f"{prefix}target_c2w has an incompatible shape")
    if batch[f"{prefix}target_intrinsics"].shape != (
        batch_size,
        frames,
        3,
        3,
    ):
        raise ValueError(f"{prefix}target_intrinsics has an incompatible shape")
    depth_name = f"{prefix}target_depth"
    if depth_name in batch and (
        batch[depth_name].ndim != 4
        or batch[depth_name].shape[:2] != (batch_size, frames)
    ):
        raise ValueError(f"{depth_name} must be aligned to latent frames")
    prefix_name = f"{prefix}prefix_condition_latents"
    if prefix_name in batch and batch[prefix_name].shape != latents.shape:
        raise ValueError(f"{prefix_name} must match {prefix}target_latents")


def validate_scene_batch(batch: dict[str, torch.Tensor]) -> None:
    missing = sorted(REQUIRED_BATCH_KEYS - set(batch))
    if missing:
        raise ValueError(f"Long-scene batch is missing keys: {missing}")
    captures = batch["capture_latents"]
    targets = batch["target_latents"]
    if captures.ndim != 5:
        raise ValueError("capture_latents must have shape (B,N,C,H,W)")
    if targets.ndim != 5:
        raise ValueError("target_latents must have shape (B,C,T,H,W)")
    batch_size, capture_frames, channels, height, width = captures.shape
    if targets.shape[:2] != (batch_size, channels):
        raise ValueError("Capture and target latent channels must match")
    if targets.shape[-2:] != (height, width):
        raise ValueError("Capture and target latent spatial sizes must match")
    if batch["capture_c2w"].shape != (batch_size, capture_frames, 4, 4):
        raise ValueError("capture_c2w has an incompatible shape")
    if batch["capture_intrinsics"].shape != (
        batch_size,
        capture_frames,
        3,
        3,
    ):
        raise ValueError("capture_intrinsics has an incompatible shape")
    if batch["capture_valid_mask"].shape != (batch_size, capture_frames):
        raise ValueError("capture_valid_mask has an incompatible shape")
    if not batch["capture_valid_mask"].bool().any(dim=1).all():
        raise ValueError("Every sample must contain at least one valid capture")
    _validate_trajectory(
        batch,
        prefix="",
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
    )
    target_frames = targets.shape[2]
    if (
        "target_write_valid_mask" in batch
        and batch["target_write_valid_mask"].shape
        != (batch_size, target_frames)
    ):
        raise ValueError(
            "target_write_valid_mask must have shape (B,target_frames)"
        )
    if batch["prompt_embeds"].ndim != 3 or (
        batch["prompt_embeds"].shape[0] != batch_size
    ):
        raise ValueError("prompt_embeds must have shape (B,L,text_dim)")
    if batch["prompt_attention_mask"].shape != batch["prompt_embeds"].shape[:2]:
        raise ValueError("prompt_attention_mask has an incompatible shape")
    optional_scene_shapes = {
        "scene_center": (batch_size, 3),
        "scene_scale": (batch_size,),
        "world_to_scene_rotation": (batch_size, 3, 3),
    }
    for name, expected_shape in optional_scene_shapes.items():
        if name in batch and batch[name].shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")

    paired_fields = {
        "paired_target_latents",
        "paired_target_c2w",
        "paired_target_intrinsics",
        "paired_target_depth",
    }
    present_paired = paired_fields & set(batch)
    if present_paired and present_paired != paired_fields:
        missing_paired = sorted(paired_fields - present_paired)
        raise ValueError(
            f"Paired-trajectory batches are missing fields: {missing_paired}"
        )
    if present_paired:
        _validate_trajectory(
            batch,
            prefix="paired_",
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
        )

    rollout_fields = {
        "rollout_target_latents",
        "rollout_target_c2w",
        "rollout_target_intrinsics",
    }
    present_rollout = rollout_fields & set(batch)
    if present_rollout and present_rollout != rollout_fields:
        missing_rollout = sorted(rollout_fields - present_rollout)
        raise ValueError(
            f"Rollout batches are missing fields: {missing_rollout}"
        )
    if present_rollout:
        _validate_trajectory(
            batch,
            prefix="rollout_",
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
        )


def load_dataset_factory(
    import_path: str,
    config_path: str | None,
) -> Dataset:
    """Load ``module:function`` without imposing a storage implementation."""

    if ":" not in import_path:
        raise ValueError("dataset_factory must use `module:function` syntax")
    module_name, function_name = import_path.split(":", 1)
    factory: Callable[[dict[str, Any]], Dataset] = getattr(
        importlib.import_module(module_name), function_name
    )
    dataset_config: dict[str, Any] = {}
    if config_path:
        with Path(config_path).open("r", encoding="utf-8") as stream:
            dataset_config = json.load(stream)
    dataset = factory(dataset_config)
    if not isinstance(dataset, Dataset):
        raise TypeError("The dataset factory must return a torch Dataset")
    return dataset


def _camera_pose(
    x: float,
    z: float,
    yaw: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    pose = torch.eye(4, dtype=dtype)
    pose[:3, :3] = torch.tensor(
        (
            (cosine, 0.0, sine),
            (0.0, 1.0, 0.0),
            (-sine, 0.0, cosine),
        ),
        dtype=dtype,
    )
    pose[:3, 3] = torch.tensor((x, 0.0, z), dtype=dtype)
    return pose


def _normalized_intrinsics(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(
        (
            (0.85, 0.0, 0.5),
            (0.0, 1.35, 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=dtype,
    )


class SyntheticLongSceneDataset(Dataset):
    """Shape-correct streaming/rollout smoke data, not scientific data."""

    def __init__(
        self,
        *,
        length: int,
        capture_frames: int,
        latent_channels: int,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        text_dim: int,
        prompt_tokens: int = 16,
    ):
        self.length = length
        self.capture_frames = capture_frames
        self.latent_channels = latent_channels
        self.latent_frames = latent_frames
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.text_dim = text_dim
        self.prompt_tokens = prompt_tokens

    def __len__(self) -> int:
        return self.length

    def _trajectory_poses(self, offset: float) -> torch.Tensor:
        return torch.stack(
            [
                _camera_pose(
                    x=0.35 * (frame + offset),
                    z=0.1 + 0.03 * math.sin(frame),
                    yaw=0.03 * frame,
                )
                for frame in range(self.latent_frames)
            ]
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(1701 + index)
        capture_latents = torch.randn(
            self.capture_frames,
            self.latent_channels,
            self.latent_height,
            self.latent_width,
            generator=generator,
        )
        capture_c2w = torch.stack(
            [
                _camera_pose(
                    x=0.35 * view,
                    z=0.15 * math.sin(view / 3.0),
                    yaw=0.05 * math.sin(view / 5.0),
                )
                for view in range(self.capture_frames)
            ]
        )
        target_c2w = self._trajectory_poses(1.5)
        rollout_c2w = self._trajectory_poses(1.5 + self.latent_frames)
        calibration = _normalized_intrinsics()
        capture_intrinsics = calibration[None].repeat(
            self.capture_frames, 1, 1
        )
        target_intrinsics = calibration[None].repeat(
            self.latent_frames, 1, 1
        )
        pairs = torch.tensor(
            [
                (pair, min(pair + 1, self.latent_frames - 1))
                for pair in range(min(4, self.latent_frames))
            ],
            dtype=torch.long,
        )
        prefix_mask = torch.zeros(
            1, self.latent_frames, 1, 1, dtype=torch.bool
        )
        prefix_mask[:, :1] = True
        return {
            "capture_latents": capture_latents,
            "capture_c2w": capture_c2w,
            "capture_intrinsics": capture_intrinsics,
            "capture_valid_mask": torch.ones(
                self.capture_frames, dtype=torch.bool
            ),
            "target_latents": torch.randn(
                self.latent_channels,
                self.latent_frames,
                self.latent_height,
                self.latent_width,
                generator=generator,
            ),
            "target_c2w": target_c2w,
            "target_intrinsics": target_intrinsics,
            "target_depth": torch.ones(
                self.latent_frames,
                self.latent_height,
                self.latent_width,
            ),
            "target_depth_confidence": torch.ones(
                self.latent_frames,
                self.latent_height,
                self.latent_width,
            ),
            "known_prefix_mask": prefix_mask,
            "target_write_valid_mask": torch.ones(
                self.latent_frames, dtype=torch.bool
            ),
            "rollout_target_latents": torch.randn(
                self.latent_channels,
                self.latent_frames,
                self.latent_height,
                self.latent_width,
                generator=generator,
            ),
            "rollout_target_c2w": rollout_c2w,
            "rollout_target_intrinsics": target_intrinsics.clone(),
            "rollout_known_prefix_mask": prefix_mask.clone(),
            "consistency_pairs": pairs,
            "consistency_pair_valid_mask": torch.ones(
                pairs.shape[0], dtype=torch.bool
            ),
            "prompt_embeds": torch.randn(
                self.prompt_tokens, self.text_dim, generator=generator
            ),
            "prompt_attention_mask": torch.ones(
                self.prompt_tokens, dtype=torch.long
            ),
        }
