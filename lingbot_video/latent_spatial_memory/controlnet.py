from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from lingbot_video.transformer_lingbot_video import make_joint_position_ids


def _patchify(tensor: torch.Tensor, patch_size: Sequence[int]) -> torch.Tensor:
    if tensor.ndim != 5:
        raise ValueError(f"expected [B,C,T,H,W], got {tuple(tensor.shape)}")
    batch, channels, frames, height, width = tensor.shape
    patch_t, patch_h, patch_w = (int(value) for value in patch_size)
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"shape {(frames, height, width)} is not divisible by patch size "
            f"{(patch_t, patch_h, patch_w)}"
        )
    grid_t, grid_h, grid_w = frames // patch_t, height // patch_h, width // patch_w
    tokens = tensor.reshape(
        batch,
        channels,
        grid_t,
        patch_t,
        grid_h,
        patch_h,
        grid_w,
        patch_w,
    )
    return tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
        batch,
        grid_t * grid_h * grid_w,
        patch_t * patch_h * patch_w * channels,
    )


class LingBotLatentMemoryControlNet(nn.Module):
    """ControlNet-style side branch initialized from selected LingBot blocks."""

    def __init__(
        self,
        *,
        hidden_size: int,
        patch_size: Sequence[int],
        block_indices: Sequence[int],
        blocks: Sequence[nn.Module],
        ray_channels: int = 6,
        segment_count: int = 3,
    ) -> None:
        super().__init__()
        if len(block_indices) != len(blocks):
            raise ValueError("block indices and side blocks must have equal length")
        self.hidden_size = int(hidden_size)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.block_indices = tuple(int(value) for value in block_indices)
        patch_volume = math.prod(self.patch_size)
        self.side_blocks = nn.ModuleList(blocks)
        self.visibility_embedder = nn.Linear(patch_volume, hidden_size)
        self.ray_embedder = nn.Linear(ray_channels * patch_volume, hidden_size)
        self.segment_embedder = nn.Embedding(segment_count, hidden_size)
        self.output_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in self.block_indices]
        )
        self.gradient_checkpointing = False
        nn.init.zeros_(self.visibility_embedder.weight)
        nn.init.zeros_(self.visibility_embedder.bias)
        nn.init.zeros_(self.ray_embedder.weight)
        nn.init.zeros_(self.ray_embedder.bias)
        for projection in self.output_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        block_indices: Sequence[int],
    ) -> "LingBotLatentMemoryControlNet":
        indices = tuple(int(value) for value in block_indices)
        if not indices or min(indices) < 0 or max(indices) >= len(backbone.blocks):
            raise ValueError(
                f"control block indices {indices} are invalid for {len(backbone.blocks)} blocks"
            )
        return cls(
            hidden_size=int(backbone.config.hidden_size),
            patch_size=backbone.config.patch_size,
            block_indices=indices,
            blocks=[copy.deepcopy(backbone.blocks[index]) for index in indices],
        )

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        *,
        backbone: nn.Module,
        noisy_latents: torch.Tensor,
        condition_latents: torch.Tensor,
        visibility: torch.Tensor,
        rays: torch.Tensor,
        segment_ids: torch.Tensor,
        target_frames: int,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        batch, channels, frames, height, width = condition_latents.shape
        if noisy_latents.shape[:2] != (batch, channels):
            raise ValueError("noisy and condition latent batch/channel shapes differ")
        if noisy_latents.shape[2:] != (target_frames, height, width):
            raise ValueError("noisy target shape does not match target condition frames")
        patch_t, patch_h, patch_w = self.patch_size
        grid_t, grid_h, grid_w = frames // patch_t, height // patch_h, width // patch_w
        target_grid_t = target_frames // patch_t
        target_tokens = target_grid_t * grid_h * grid_w
        if visibility.shape != (batch, 1, frames, height, width):
            raise ValueError("visibility shape does not match condition latents")
        if rays.shape != (batch, 6, frames, height, width):
            raise ValueError("ray shape does not match condition latents")
        if segment_ids.shape != (batch, frames):
            raise ValueError("segment_ids must be [B,T]")

        latent_tokens = _patchify(condition_latents, self.patch_size)
        condition = backbone.patch_embedder(latent_tokens)
        # A ControlNet branch must remain aware of the current denoising state.
        # The aligned memory uses the same native latent channels and therefore
        # the same shared patch embedder, without a learned bridging encoder.
        noisy_tokens = backbone.patch_embedder(
            _patchify(noisy_latents, self.patch_size)
        )
        memory_tokens = condition[:, :target_tokens]
        if backbone.patch_embedder.bias is not None:
            # Zero-filled unseen memory must contribute zero, rather than a
            # second copy of the shared patch-embedding bias.
            memory_tokens = memory_tokens - backbone.patch_embedder.bias
        condition = torch.cat(
            (
                memory_tokens + noisy_tokens,
                condition[:, target_tokens:],
            ),
            dim=1,
        )
        condition = condition + self.visibility_embedder(
            _patchify(visibility, self.patch_size).to(condition.dtype)
        )
        condition = condition + self.ray_embedder(
            _patchify(rays, self.patch_size).to(condition.dtype)
        )
        patch_segments = segment_ids[:, ::patch_t]
        segment_tokens = patch_segments[:, :, None].expand(-1, -1, grid_h * grid_w)
        segment_tokens = segment_tokens.reshape(batch, -1)
        condition = condition + self.segment_embedder(segment_tokens).to(condition.dtype)

        text = backbone.text_embedder(encoder_hidden_states)
        joint = torch.cat((condition, text), dim=1)
        text_length = encoder_hidden_states.shape[1]
        if encoder_attention_mask is None:
            text_lens = torch.full(
                (batch,),
                text_length,
                device=joint.device,
                dtype=torch.long,
            )
        else:
            text_lens = encoder_attention_mask.sum(dim=-1).long()
        rotary = torch.stack(
            [
                backbone.rope(
                    make_joint_position_ids(
                        text_length,
                        grid_t,
                        grid_h,
                        grid_w,
                        joint.device,
                        patch_segments[index],
                    )
                )
                for index in range(batch)
            ],
            dim=0,
        )
        attention_mask = None
        moe_padding_mask = None
        if encoder_attention_mask is not None and bool(
            (text_lens < text_length).any().item()
        ):
            key_mask = torch.cat(
                (
                    torch.ones(
                        batch,
                        condition.shape[1],
                        device=joint.device,
                        dtype=torch.bool,
                    ),
                    encoder_attention_mask.bool(),
                ),
                dim=1,
            )
            attention_mask = key_mask[:, None, None, :]
            moe_padding_mask = key_mask.reshape(-1).float()

        timestep_projection = backbone.time_proj(timestep.float())
        timestep_embedding = backbone.time_embedder(timestep_projection)
        timestep_input = timestep_embedding[:, None, :].expand(
            batch,
            joint.shape[1],
            -1,
        )
        temb6 = backbone.time_modulation(
            timestep_input.reshape(batch * joint.shape[1], -1)
        )

        residuals: dict[int, torch.Tensor] = {}
        for output_layer, side_block, output_projection in zip(
            self.block_indices,
            self.side_blocks,
            self.output_projections,
            strict=True,
        ):
            if self.training and self.gradient_checkpointing and torch.is_grad_enabled():
                def custom_forward(
                    hidden_states: torch.Tensor,
                    modulation: torch.Tensor,
                    rotary_embedding: torch.Tensor,
                    current_block: nn.Module = side_block,
                ) -> torch.Tensor:
                    return current_block(
                        hidden_states,
                        modulation,
                        rotary_embedding,
                        attention_mask,
                        moe_padding_mask,
                    )

                joint = checkpoint(
                    custom_forward,
                    joint,
                    temb6,
                    rotary,
                    use_reentrant=False,
                )
            else:
                joint = side_block(
                    joint,
                    temb6,
                    rotary,
                    attention_mask,
                    moe_padding_mask,
                )
            residuals[output_layer] = output_projection(
                joint[:, :target_tokens]
            )
        return residuals
