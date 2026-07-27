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
    """MIRAGE/VACE side branch adapted to LingBot's joint-attention blocks.

    The topology follows Appendix C and VACE: latent-memory tokens use the
    backbone's native patch embedding, the first side block receives
    ``before_proj(memory) + x``, side blocks run recurrently, and zero-initialized
    ``after_proj`` layers inject hints after the selected backbone layers.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        patch_size: Sequence[int],
        block_indices: Sequence[int],
        blocks: Sequence[nn.Module],
        ray_channels: int = 6,
    ) -> None:
        super().__init__()
        if len(block_indices) != len(blocks):
            raise ValueError("block indices and side blocks must have equal length")
        if not block_indices:
            raise ValueError("at least one ControlNet block is required")
        self.hidden_size = int(hidden_size)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.block_indices = tuple(int(value) for value in block_indices)
        patch_volume = math.prod(self.patch_size)

        self.side_blocks = nn.ModuleList(blocks)
        self.before_projection = nn.Linear(hidden_size, hidden_size)
        self.after_projections = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in self.block_indices]
        )
        # The paper explicitly returns a visibility mask.  The memory latent
        # still goes through the shared C-channel patch embedding; these
        # zero-initialized adapters add mask and pose information without a
        # learned latent-space bridge.
        self.visibility_embedder = nn.Linear(patch_volume, hidden_size)
        self.ray_embedder = nn.Linear(ray_channels * patch_volume, hidden_size)
        self.gradient_checkpointing = False

        nn.init.zeros_(self.visibility_embedder.weight)
        nn.init.zeros_(self.visibility_embedder.bias)
        nn.init.zeros_(self.ray_embedder.weight)
        nn.init.zeros_(self.ray_embedder.bias)
        for projection in self.after_projections:
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    @classmethod
    def from_backbone(
        cls,
        backbone: nn.Module,
        block_indices: Sequence[int],
    ) -> "LingBotLatentMemoryControlNet":
        indices = tuple(int(value) for value in block_indices)
        if (
            not indices
            or tuple(sorted(set(indices))) != indices
            or min(indices) < 0
            or max(indices) >= len(backbone.blocks)
        ):
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

    @staticmethod
    def _token_time_modulation(
        backbone: nn.Module,
        frame_timesteps: torch.Tensor,
        *,
        grid_h: int,
        grid_w: int,
        text_length: int,
    ) -> torch.Tensor:
        batch, frames = frame_timesteps.shape
        spatial_tokens = grid_h * grid_w
        video_timestep = frame_timesteps[:, :, None].expand(
            batch,
            frames,
            spatial_tokens,
        ).reshape(batch, frames * spatial_tokens)
        text_timestep = frame_timesteps.amax(dim=1, keepdim=True).expand(
            batch,
            text_length,
        )
        token_timestep = torch.cat((video_timestep, text_timestep), dim=1)
        flat = token_timestep.reshape(-1).float()
        embedding = backbone.time_embedder(backbone.time_proj(flat))
        return backbone.time_modulation(embedding)

    def forward(
        self,
        *,
        backbone: nn.Module,
        backbone_latents: torch.Tensor,
        memory_latents: torch.Tensor,
        visibility: torch.Tensor,
        rays: torch.Tensor,
        segment_ids: torch.Tensor,
        frame_timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        batch, channels, frames, height, width = memory_latents.shape
        expected_shape = (batch, channels, frames, height, width)
        if backbone_latents.shape != expected_shape:
            raise ValueError(
                "backbone/control latent shapes differ: "
                f"{tuple(backbone_latents.shape)} vs {expected_shape}"
            )
        if visibility.shape != (batch, 1, frames, height, width):
            raise ValueError("visibility shape does not match memory latents")
        if rays.shape != (batch, 6, frames, height, width):
            raise ValueError("ray shape does not match memory latents")
        if segment_ids.shape != (batch, frames):
            raise ValueError("segment_ids must be [B,T]")
        if frame_timesteps.shape != (batch, frames):
            raise ValueError("frame_timesteps must be [B,T]")

        patch_t, patch_h, patch_w = self.patch_size
        grid_t = frames // patch_t
        grid_h = height // patch_h
        grid_w = width // patch_w
        if grid_t != frames:
            raise NotImplementedError(
                "frame-wise MIRAGE conditioning currently requires temporal patch size 1"
            )

        memory_tokens = backbone.patch_embedder(
            _patchify(memory_latents, self.patch_size)
        )
        if backbone.patch_embedder.bias is not None:
            # Zero-filled unseen memory must stay zero before the explicit mask
            # embedding is added.
            memory_tokens = memory_tokens - backbone.patch_embedder.bias
        memory_tokens = memory_tokens + self.visibility_embedder(
            _patchify(visibility, self.patch_size).to(memory_tokens.dtype)
        )
        memory_tokens = memory_tokens + self.ray_embedder(
            _patchify(rays, self.patch_size).to(memory_tokens.dtype)
        )
        backbone_tokens = backbone.patch_embedder(
            _patchify(backbone_latents, self.patch_size)
        )
        control = self.before_projection(memory_tokens) + backbone_tokens

        text = backbone.text_embedder(encoder_hidden_states)
        joint = torch.cat((control, text), dim=1)
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

        patch_segments = segment_ids[:, ::patch_t]
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
                        control.shape[1],
                        device=joint.device,
                        dtype=torch.bool,
                    ),
                    encoder_attention_mask.bool(),
                ),
                dim=1,
            )
            attention_mask = key_mask[:, None, None, :]
            moe_padding_mask = key_mask.reshape(-1).float()

        temb6 = self._token_time_modulation(
            backbone,
            frame_timesteps,
            grid_h=grid_h,
            grid_w=grid_w,
            text_length=text_length,
        )

        residuals: dict[int, torch.Tensor] = {}
        video_tokens = control.shape[1]
        for output_layer, side_block, after_projection in zip(
            self.block_indices,
            self.side_blocks,
            self.after_projections,
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
            residuals[output_layer] = after_projection(joint[:, :video_tokens])
        return residuals
