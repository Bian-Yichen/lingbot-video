from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

from .geometry import camera_vector, make_origin_direction_rays
from .memory_encoder import (
    CameraQueryableGeometryHead,
    GIMImplicitMemoryEncoder,
    GIMMemoryEncoderConfig,
    TargetCameraActionEncoder,
)


@dataclass(frozen=True)
class GIMWorldModelConfig:
    image_height: int = 480
    image_width: int = 832
    vae_spatial_stride: int = 8
    memory_latent_frames: int = 20
    memory_depth: int = 2
    compact_stride: int = 2
    memory_intermediate_ratio: float = 4.0
    teacher_grid_height: int = 21
    teacher_grid_width: int = 37
    teacher_feature_dim: int = 2048
    geometry_num_heads: int = 16

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GIMWorldLingBotModel(nn.Module):
    """Paper-faithful GIM modules attached to the LingBot diffusion backbone."""

    def __init__(
        self,
        backbone: LingBotVideoTransformer3DModel,
        config: GIMWorldModelConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.gim_config = config
        hidden_size = int(backbone.config.hidden_size)
        patch_t, patch_h, patch_w = tuple(backbone.config.patch_size)
        if patch_t != 1:
            raise ValueError("GIM history patchification currently requires patch_t=1")
        latent_h = config.image_height // config.vae_spatial_stride
        latent_w = config.image_width // config.vae_spatial_stride
        if latent_h % patch_h or latent_w % patch_w:
            raise ValueError("latent resolution is not divisible by DiT patch size")
        grid_h, grid_w = latent_h // patch_h, latent_w // patch_w
        memory_cfg = GIMMemoryEncoderConfig(
            hidden_size=hidden_size,
            num_heads=int(backbone.config.num_attention_heads),
            intermediate_size=int(hidden_size * config.memory_intermediate_ratio),
            memory_latent_frames=config.memory_latent_frames,
            patch_height=grid_h,
            patch_width=grid_w,
            compact_stride=config.compact_stride,
            depth=config.memory_depth,
            camera_input_dim=16,
            norm_eps=float(backbone.config.norm_eps),
            axes_dims=tuple(backbone.config.axes_dims),
            axes_lens=tuple(backbone.config.axes_lens),
            rope_theta=float(backbone.config.rope_theta),
        )
        self.memory_encoder = GIMImplicitMemoryEncoder(memory_cfg)
        self.geometry_head = CameraQueryableGeometryHead(
            hidden_size=hidden_size,
            teacher_dim=config.teacher_feature_dim,
            grid_height=config.teacher_grid_height,
            grid_width=config.teacher_grid_width,
            num_heads=config.geometry_num_heads,
            intermediate_size=hidden_size * 4,
            norm_eps=float(backbone.config.norm_eps),
        )
        self.action_encoder = TargetCameraActionEncoder(16, hidden_size)

    @property
    def patch_grid(self) -> tuple[int, int]:
        cfg = self.memory_encoder.config
        return cfg.patch_height, cfg.patch_width

    def patchify_history(self, latents: torch.Tensor) -> torch.Tensor:
        """Use the *shared backbone patch embedding* as required by section 3.2."""

        batch, channels, frames, height, width = latents.shape
        patch_t, patch_h, patch_w = tuple(self.backbone.config.patch_size)
        grid_t = frames // patch_t
        grid_h = height // patch_h
        grid_w = width // patch_w
        patches = (
            latents.reshape(
                batch,
                channels,
                grid_t,
                patch_t,
                grid_h,
                patch_h,
                grid_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(
                batch,
                grid_t,
                grid_h * grid_w,
                patch_t * patch_h * patch_w * channels,
            )
        )
        return self.backbone.patch_embedder(patches)

    def build_memory(
        self,
        history_latents: torch.Tensor,
        history_c2w: torch.Tensor,
        history_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.patchify_history(history_latents)
        cameras = camera_vector(
            history_c2w,
            history_intrinsics,
            (
                self.gim_config.image_height,
                self.gim_config.image_width,
            ),
        )
        return self.memory_encoder(tokens, cameras)

    def target_action_embeddings(
        self,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        cameras = camera_vector(
            target_c2w,
            target_intrinsics,
            (
                self.gim_config.image_height,
                self.gim_config.image_width,
            ),
        )
        return self.action_encoder(cameras)

    def geometry_prediction(
        self,
        memory: torch.Tensor,
        query_c2w: torch.Tensor,
        query_intrinsics: torch.Tensor,
        *,
        teacher_image_hw: tuple[int, int],
    ) -> torch.Tensor:
        teacher_h, teacher_w = teacher_image_hw
        image_h, image_w = (
            self.gim_config.image_height,
            self.gim_config.image_width,
        )
        scaled_k = query_intrinsics.clone()
        scaled_k[..., 0, 0] *= teacher_w / image_w
        scaled_k[..., 1, 1] *= teacher_h / image_h
        scaled_k[..., 0, 2] = (
            (scaled_k[..., 0, 2] + 0.5) * teacher_w / image_w - 0.5
        )
        scaled_k[..., 1, 2] = (
            (scaled_k[..., 1, 2] + 0.5) * teacher_h / image_h - 0.5
        )
        ray_h = self.gim_config.teacher_grid_height
        ray_w = self.gim_config.teacher_grid_width
        scaled_k[..., 0, 0] *= ray_w / teacher_w
        scaled_k[..., 1, 1] *= ray_h / teacher_h
        scaled_k[..., 0, 2] = (
            (scaled_k[..., 0, 2] + 0.5) * ray_w / teacher_w - 0.5
        )
        scaled_k[..., 1, 2] = (
            (scaled_k[..., 1, 2] + 0.5) * ray_h / teacher_h - 0.5
        )
        rays = make_origin_direction_rays(
            query_c2w,
            scaled_k,
            ray_h,
            ray_w,
        )
        return self.geometry_head(memory, rays)

    def denoise(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        memory: torch.Tensor,
        target_action_embeddings: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.backbone(
            noisy_latents,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            memory_hidden_states=memory,
            video_action_embeds=target_action_embeddings,
            return_dict=False,
        )[0]

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        history_latents: torch.Tensor,
        history_c2w: torch.Tensor,
        history_intrinsics: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
        query_c2w: torch.Tensor,
        query_intrinsics: torch.Tensor,
        teacher_image_hw: tuple[int, int],
        compute_geometry: bool = True,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Joint paper training path, kept in one forward for DDP/FSDP."""

        memory = self.build_memory(
            history_latents,
            history_c2w,
            history_intrinsics,
        )
        actions = self.target_action_embeddings(
            target_c2w,
            target_intrinsics,
        )
        prediction = self.denoise(
            noisy_latents,
            timestep,
            encoder_hidden_states,
            memory=memory,
            target_action_embeddings=actions,
            encoder_attention_mask=encoder_attention_mask,
        )
        geometry = None
        if compute_geometry:
            geometry = self.geometry_prediction(
                memory,
                query_c2w,
                query_intrinsics,
                teacher_image_hw=teacher_image_hw,
            )
        return prediction, geometry, memory
