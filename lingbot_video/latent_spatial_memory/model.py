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
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        memory_latents: torch.Tensor,
        memory_visibility: torch.Tensor,
        target_rays: torch.Tensor,
        target_segment_ids: torch.Tensor,
        reference_latents: torch.Tensor,
        reference_rays: torch.Tensor,
    ) -> LatentMemoryModelOutput:
        batch, _channels, target_frames, height, width = noisy_latents.shape
        reference_frames = reference_latents.shape[2]
        condition_latents = torch.cat((memory_latents, reference_latents), dim=2)
        condition_visibility = torch.cat(
            (
                memory_visibility,
                torch.ones(
                    batch,
                    1,
                    reference_frames,
                    height,
                    width,
                    device=memory_visibility.device,
                    dtype=memory_visibility.dtype,
                ),
            ),
            dim=2,
        )
        condition_rays = torch.cat((target_rays, reference_rays), dim=2)
        condition_segments = torch.cat(
            (
                target_segment_ids,
                torch.full(
                    (batch, reference_frames),
                    REFERENCE_SEGMENT,
                    device=target_segment_ids.device,
                    dtype=target_segment_ids.dtype,
                ),
            ),
            dim=1,
        )
        residuals = self.controlnet(
            backbone=self.backbone,
            noisy_latents=noisy_latents,
            condition_latents=condition_latents,
            visibility=condition_visibility,
            rays=condition_rays,
            segment_ids=condition_segments,
            target_frames=target_frames,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        velocity = self.backbone(
            noisy_latents,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            video_segment_ids=target_segment_ids,
            block_control_residuals=residuals,
            return_dict=False,
        )[0]
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
