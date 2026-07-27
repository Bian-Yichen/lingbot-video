from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .controlnet import LingBotLatentMemoryControlNet


TARGET_SEGMENT = 0
PRECEDING_SEGMENT = 1
REFERENCE_SEGMENT = 2


@dataclass
class LatentMemoryModelOutput:
    velocity: torch.Tensor
    control_residuals: dict[int, torch.Tensor]


class LatentMetricDepthHead(nn.Module):
    """Predict metric log-depth for generated latents without an external model."""

    def __init__(self, latent_channels: int, hidden_channels: int = 128) -> None:
        super().__init__()
        input_channels = latent_channels + 6 + 2
        groups = min(32, hidden_channels)
        while hidden_channels % groups:
            groups -= 1
        self.network = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, 1, kernel_size=1),
        )

    def forward(
        self,
        clean_latents: torch.Tensor,
        rays: torch.Tensor,
        projected_depth: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        depth_hint = torch.log1p(projected_depth.clamp_min(0)) * visibility
        inputs = torch.cat(
            (
                clean_latents,
                rays.to(clean_latents),
                depth_hint.to(clean_latents),
                visibility.to(clean_latents),
            ),
            dim=1,
        )
        return self.network(inputs).float()


class LingBotVideoLatentMemoryModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        controlnet: LingBotLatentMemoryControlNet,
        depth_head: LatentMetricDepthHead,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.controlnet = controlnet
        self.depth_head = depth_head

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "enable_gradient_checkpointing"):
            self.backbone.enable_gradient_checkpointing()
        self.controlnet.enable_gradient_checkpointing()

    def forward(
        self,
        *,
        noisy_latents: torch.Tensor,
        target_timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        memory_latents: torch.Tensor,
        memory_visibility: torch.Tensor,
        target_rays: torch.Tensor,
        preceding_latents: torch.Tensor,
        preceding_rays: torch.Tensor,
        reference_latents: torch.Tensor,
    ) -> LatentMemoryModelOutput:
        batch, _channels, target_frames, height, width = noisy_latents.shape
        preceding_frames = preceding_latents.shape[2]
        reference_frames = reference_latents.shape[2]
        conditioned_frames = target_frames + preceding_frames
        if memory_latents.shape != (
            batch,
            noisy_latents.shape[1],
            conditioned_frames,
            height,
            width,
        ):
            raise ValueError(
                "memory_latents must align with [target, preceding] frames"
            )
        if memory_visibility.shape != (
            batch,
            1,
            conditioned_frames,
            height,
            width,
        ):
            raise ValueError("memory_visibility must align with memory_latents")
        if target_timesteps.shape != (batch, target_frames):
            raise ValueError("target_timesteps must be [B,T_target]")

        control_latents = torch.cat((noisy_latents, preceding_latents), dim=2)
        control_rays = torch.cat((target_rays, preceding_rays), dim=2)
        control_segments = torch.cat(
            (
                torch.full(
                    (batch, target_frames),
                    TARGET_SEGMENT,
                    device=noisy_latents.device,
                    dtype=torch.long,
                ),
                torch.full(
                    (batch, preceding_frames),
                    PRECEDING_SEGMENT,
                    device=noisy_latents.device,
                    dtype=torch.long,
                ),
            ),
            dim=1,
        )
        control_timesteps = torch.cat(
            (
                target_timesteps,
                torch.zeros(
                    batch,
                    preceding_frames,
                    device=target_timesteps.device,
                    dtype=target_timesteps.dtype,
                ),
            ),
            dim=1,
        )
        residuals = self.controlnet(
            backbone=self.backbone,
            backbone_latents=control_latents,
            memory_latents=memory_latents,
            visibility=memory_visibility,
            rays=control_rays,
            segment_ids=control_segments,
            frame_timesteps=control_timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # MIRAGE performs one forward over [reference, noisy target, preceding].
        # Clean context uses timestep zero and only the target slice is trained.
        backbone_latents = torch.cat(
            (reference_latents, noisy_latents, preceding_latents),
            dim=2,
        )
        backbone_timesteps = torch.cat(
            (
                torch.zeros(
                    batch,
                    reference_frames,
                    device=target_timesteps.device,
                    dtype=target_timesteps.dtype,
                ),
                target_timesteps,
                torch.zeros(
                    batch,
                    preceding_frames,
                    device=target_timesteps.device,
                    dtype=target_timesteps.dtype,
                ),
            ),
            dim=1,
        )
        backbone_segments = torch.cat(
            (
                torch.full(
                    (batch, reference_frames),
                    REFERENCE_SEGMENT,
                    device=noisy_latents.device,
                    dtype=torch.long,
                ),
                torch.full(
                    (batch, target_frames),
                    TARGET_SEGMENT,
                    device=noisy_latents.device,
                    dtype=torch.long,
                ),
                torch.full(
                    (batch, preceding_frames),
                    PRECEDING_SEGMENT,
                    device=noisy_latents.device,
                    dtype=torch.long,
                ),
            ),
            dim=1,
        )
        velocity_all = self.backbone(
            backbone_latents,
            backbone_timesteps,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            video_segment_ids=backbone_segments,
            block_control_residuals=residuals,
            block_control_frame_range=(
                reference_frames,
                reference_frames + conditioned_frames,
            ),
            return_dict=False,
        )[0]
        velocity = velocity_all[
            :,
            :,
            reference_frames : reference_frames + target_frames,
        ]
        return LatentMemoryModelOutput(velocity=velocity, control_residuals=residuals)

    def predict_log_depth(
        self,
        clean_latents: torch.Tensor,
        target_rays: torch.Tensor,
        projected_depth: torch.Tensor,
        memory_visibility: torch.Tensor,
    ) -> torch.Tensor:
        return self.depth_head(
            clean_latents,
            target_rays,
            projected_depth,
            memory_visibility,
        )


def metric_depth_loss(
    predicted_log_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    absolute_weight: float = 0.1,
) -> torch.Tensor:
    """Scale-invariant log-depth plus a metric-scale robust term."""

    valid = (
        valid_mask.bool()
        & torch.isfinite(target_depth)
        & (target_depth > 0)
    )
    if not valid.any():
        return predicted_log_depth.sum() * 0
    target_log = torch.log(target_depth.clamp_min(1e-6))
    residual = predicted_log_depth - target_log
    selected = residual[valid]
    scale_invariant = (selected - selected.mean()).square().mean()
    metric = F.smooth_l1_loss(selected, torch.zeros_like(selected), beta=0.1)
    return scale_invariant + absolute_weight * metric
