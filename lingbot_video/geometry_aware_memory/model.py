from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel

from .geometry import camera_vector
from .memory_encoder import TargetCameraActionEncoder


@dataclass(frozen=True)
class GIMWorldModelConfig:
    image_height: int = 480
    image_width: int = 832
    vae_spatial_stride: int = 8

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
        self._patch_grid = (grid_h, grid_w)
        self.action_encoder = TargetCameraActionEncoder(16, hidden_size)

    @property
    def patch_grid(self) -> tuple[int, int]:
        return self._patch_grid

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
        del history_c2w, history_intrinsics
        return self.patchify_history(history_latents)

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
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow-matching path with uncompressed history tokens."""

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
        return prediction, memory
