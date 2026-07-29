from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VGGTTeacherConfig:
    model_id: str = "facebook/VGGT-1B"
    input_width: int = 518
    patch_size: int = 14
    feature_dim: int = 2048
    cache_dir: Optional[str] = None


def vggt_target_hw(
    source_hw: tuple[int, int],
    target_width: int = 518,
    patch_size: int = 14,
) -> tuple[int, int]:
    source_h, source_w = source_hw
    target_h = round(source_h * (target_width / source_w) / patch_size) * patch_size
    target_h = min(target_width, max(patch_size, target_h))
    return int(target_h), int(target_width)


def preprocess_vggt_tensor(
    image: torch.Tensor,
    config: VGGTTeacherConfig,
) -> torch.Tensor:
    """Tensor equivalent of VGGT's official ``load_and_preprocess_images`` crop mode."""

    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("VGGT image must be [B,3,H,W]")
    target_hw = vggt_target_hw(
        (int(image.shape[-2]), int(image.shape[-1])),
        config.input_width,
        config.patch_size,
    )
    resized = F.interpolate(
        image.float(),
        size=target_hw,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    if resized.shape[-2] > config.input_width:
        top = (resized.shape[-2] - config.input_width) // 2
        resized = resized[..., top : top + config.input_width, :]
    return resized.clamp(0, 1)


class VGGTGeometryTeacher:
    """Frozen official VGGT encoder used only for GIM geometry distillation."""

    def __init__(
        self,
        config: VGGTTeacherConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise RuntimeError(
                "GIM-World training requires the official VGGT package. "
                "Install requirements-gim-world.txt before training."
            ) from exc
        self.config = config
        self.device = device
        self.dtype = dtype
        self.model = VGGT.from_pretrained(config.model_id)
        # GIM supervises encoder patch features only; VGGT prediction heads are
        # neither called nor needed in memory.
        self.model.camera_head = None
        self.model.depth_head = None
        self.model.point_head = None
        self.model.track_head = None
        self.model.requires_grad_(False)
        self.model.eval().to(device)
        self.cache_dir = Path(config.cache_dir) if config.cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(
        self,
        item_name: str,
        frame_index: int,
        input_hw: tuple[int, int],
    ) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        safe_model = self.config.model_id.replace("/", "--")
        height, width = input_hw
        directory = self.cache_dir / safe_model / item_name
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{frame_index:06d}_{height}x{width}.pt"

    @torch.no_grad()
    def encode(
        self,
        image: torch.Tensor,
        *,
        item_name: str,
        frame_index: int,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        cache_path = self._cache_path(
            item_name,
            frame_index,
            (int(image.shape[-2]), int(image.shape[-1])),
        )
        if cache_path is not None and cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            return payload["features"].to(self.device), tuple(payload["grid_hw"])

        teacher_image = preprocess_vggt_tensor(image, self.config).to(self.device)
        autocast = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            and self.dtype in {torch.float16, torch.bfloat16}
            else contextlib.nullcontext()
        )
        with autocast:
            outputs, patch_start = self.model.aggregator(teacher_image.unsqueeze(1))
        final_tokens = next(value for value in reversed(outputs) if value is not None)
        features = final_tokens[:, :, patch_start:].squeeze(1).float()
        grid_hw = (
            teacher_image.shape[-2] // self.config.patch_size,
            teacher_image.shape[-1] // self.config.patch_size,
        )
        if features.shape[1] != grid_hw[0] * grid_hw[1]:
            raise RuntimeError(
                f"VGGT returned {features.shape[1]} patches for grid {grid_hw}"
            )
        if cache_path is not None:
            torch.save(
                {
                    "features": features.cpu().to(torch.float16),
                    "grid_hw": grid_hw,
                },
                cache_path,
            )
        return features, grid_hw
