from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from lingbot_video.transformer_lingbot_video import (
    LingBotVideoRMSNorm,
    LingBotVideoRotaryEmbedding,
    apply_rotary_emb,
)


def _grid_position_ids(
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    tt = torch.arange(frames, device=device, dtype=torch.int32)
    yy = torch.arange(height, device=device, dtype=torch.int32)
    xx = torch.arange(width, device=device, dtype=torch.int32)
    return torch.stack(
        torch.meshgrid(tt, yy, xx, indexing="ij"),
        dim=-1,
    ).flatten(0, 2)


class CompactLinear(nn.Module):
    """Paper equations (9)-(10): shared 2D block flatten/linear transform."""

    def __init__(self, hidden_size: int, stride: int, *, expand: bool) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.stride = int(stride)
        area = self.stride * self.stride
        self.projection = (
            nn.Linear(hidden_size, area * hidden_size)
            if expand
            else nn.Linear(area * hidden_size, hidden_size)
        )
        self.expand = bool(expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B,F,H,W,D].
        batch, frames, height, width, dim = x.shape
        stride = self.stride
        if height % stride or width % stride:
            raise ValueError(
                f"patch grid {height}x{width} is not divisible by compact stride {stride}"
            )
        if self.expand:
            projected = self.projection(x)
            projected = projected.reshape(
                batch,
                frames,
                height,
                width,
                stride,
                stride,
                dim,
            )
            return (
                projected.permute(0, 1, 2, 4, 3, 5, 6)
                .reshape(
                    batch,
                    frames,
                    height * stride,
                    width * stride,
                    dim,
                )
            )
        return self.projection(
            x.reshape(
                batch,
                frames,
                height // stride,
                stride,
                width // stride,
                stride,
                dim,
            )
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(
                batch,
                frames,
                height // stride,
                width // stride,
                stride * stride * dim,
            )
        )


class CompactSelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        norm_eps: float,
        axes_dims: tuple[int, int, int],
        axes_lens: tuple[int, int, int],
        rope_theta: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        if sum(axes_dims) != self.head_dim:
            raise ValueError(
                f"memory axes_dims sum {sum(axes_dims)} != head_dim {self.head_dim}"
            )
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.rope = LingBotVideoRotaryEmbedding(
            axes_dims,
            axes_lens,
            rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, hidden = x.shape
        q = self.to_q(x).reshape(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        )
        k = self.to_k(x).reshape_as(q)
        v = self.to_v(x).reshape_as(q)
        rotary = self.rope(position_ids)
        q = apply_rotary_emb(self.norm_q(q), rotary)
        k = apply_rotary_emb(self.norm_k(k), rotary)
        output = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).reshape(batch, length, hidden)
        return self.to_out(output)


class GIMMemoryEncoderBlock(nn.Module):
    """Equations (7)-(8), with attention compacted but FFN full resolution."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        compact_stride: int,
        norm_eps: float,
        axes_dims: tuple[int, int, int],
        axes_lens: tuple[int, int, int],
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.compact = CompactLinear(
            hidden_size,
            compact_stride,
            expand=False,
        )
        self.expand = CompactLinear(
            hidden_size,
            compact_stride,
            expand=True,
        )
        self.attention_norm = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.attention = CompactSelfAttention(
            hidden_size,
            num_heads,
            norm_eps,
            axes_dims,
            axes_lens,
            rope_theta,
        )
        self.ffn_norm = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )

    def forward(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Both tensors are [B,F,H,W,D].  Compact and Expand are applied by the
        # same projections to both segments as required by the paper.
        q_compact = self.compact(self.attention_norm(query))
        h_compact = self.compact(self.attention_norm(history))
        batch, query_frames, hc, wc, dim = q_compact.shape
        history_frames = h_compact.shape[1]
        packed = torch.cat(
            (
                q_compact.flatten(1, 3),
                h_compact.flatten(1, 3),
            ),
            dim=1,
        )
        # Separate query/history rotary grids: neither segment inherits the
        # other's temporal coordinate range.
        position_ids = torch.cat(
            (
                _grid_position_ids(query_frames, hc, wc, packed.device),
                _grid_position_ids(history_frames, hc, wc, packed.device),
            ),
            dim=0,
        )
        attended = self.attention(packed, position_ids)
        query_length = query_frames * hc * wc
        q_attended = attended[:, :query_length].reshape(
            batch,
            query_frames,
            hc,
            wc,
            dim,
        )
        h_attended = attended[:, query_length:].reshape(
            batch,
            history_frames,
            hc,
            wc,
            dim,
        )
        query = query + self.expand(q_attended)
        history = history + self.expand(h_attended)
        query = query + self.ffn(self.ffn_norm(query))
        history = history + self.ffn(self.ffn_norm(history))
        return query, history


@dataclass(frozen=True)
class GIMMemoryEncoderConfig:
    hidden_size: int
    num_heads: int
    intermediate_size: int
    memory_latent_frames: int
    patch_height: int
    patch_width: int
    compact_stride: int = 2
    depth: int = 2
    camera_input_dim: int = 16
    norm_eps: float = 1e-6
    axes_dims: tuple[int, int, int] = (32, 48, 48)
    axes_lens: tuple[int, int, int] = (8192, 1024, 1024)
    rope_theta: float = 256.0


class GIMImplicitMemoryEncoder(nn.Module):
    """Fixed-size geometry-aware implicit memory from GIM-World section 3.2."""

    def __init__(self, config: GIMMemoryEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        if config.depth != 2:
            raise ValueError("the paper specifies exactly two memory blocks")
        if (
            config.patch_height % config.compact_stride
            or config.patch_width % config.compact_stride
        ):
            raise ValueError("memory patch grid must be divisible by compact stride")
        query_count = (
            config.memory_latent_frames
            * config.patch_height
            * config.patch_width
        )
        self.memory_queries = nn.Parameter(
            torch.empty(1, query_count, config.hidden_size)
        )
        nn.init.normal_(self.memory_queries, std=0.02)
        self.camera_embedding = nn.Linear(
            config.camera_input_dim,
            config.hidden_size,
        )
        self.blocks = nn.ModuleList(
            [
                GIMMemoryEncoderBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    intermediate_size=config.intermediate_size,
                    compact_stride=config.compact_stride,
                    norm_eps=config.norm_eps,
                    axes_dims=config.axes_dims,
                    axes_lens=config.axes_lens,
                    rope_theta=config.rope_theta,
                )
                for _ in range(config.depth)
            ]
        )
        self.output_norm = LingBotVideoRMSNorm(
            config.hidden_size,
            config.norm_eps,
        )

    @property
    def memory_token_count(self) -> int:
        return int(self.memory_queries.shape[1])

    def forward(
        self,
        history_tokens: torch.Tensor,
        history_camera_vectors: torch.Tensor,
    ) -> torch.Tensor:
        # history_tokens: [B,T,P,D], one patch grid per retained latent frame.
        batch, frames, patches, dim = history_tokens.shape
        cfg = self.config
        if patches != cfg.patch_height * cfg.patch_width:
            raise ValueError(
                f"history has {patches} patches/frame but memory expects "
                f"{cfg.patch_height}x{cfg.patch_width}"
            )
        if history_camera_vectors.shape[:2] != (batch, frames):
            raise ValueError("camera vector count must match history frames")
        camera = self.camera_embedding(history_camera_vectors).unsqueeze(2)
        history = (history_tokens + camera).reshape(
            batch,
            frames,
            cfg.patch_height,
            cfg.patch_width,
            dim,
        )
        query = self.memory_queries.to(history_tokens.dtype).expand(
            batch,
            -1,
            -1,
        ).reshape(
            batch,
            cfg.memory_latent_frames,
            cfg.patch_height,
            cfg.patch_width,
            dim,
        )
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                query, history = checkpoint(
                    block,
                    query,
                    history,
                    use_reentrant=False,
                )
            else:
                query, history = block(query, history)
        del history
        return self.output_norm(query).flatten(1, 3)


class CameraQueryableGeometryHead(nn.Module):
    """Paper section 3.3: ray queries -> memory cross-attn -> patch self-attn."""

    def __init__(
        self,
        hidden_size: int,
        teacher_dim: int,
        grid_height: int,
        grid_width: int,
        num_heads: int,
        intermediate_size: int,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.grid_height = int(grid_height)
        self.grid_width = int(grid_width)
        self.ray_mlp = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.grid_embedding = nn.Parameter(
            torch.empty(
                1,
                self.grid_height * self.grid_width,
                hidden_size,
            )
        )
        nn.init.normal_(self.grid_embedding, std=0.02)
        self.query_norm = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.memory_norm = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.self_norm = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.self_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            batch_first=True,
        )
        self.ffn_norm = LingBotVideoRMSNorm(hidden_size, norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.output_projection = nn.Linear(hidden_size, teacher_dim)

    def forward(
        self,
        memory: torch.Tensor,
        rays: torch.Tensor,
    ) -> torch.Tensor:
        batch = memory.shape[0]
        rays = rays.reshape(batch, self.grid_height * self.grid_width, 6)
        query = self.ray_mlp(rays) + self.grid_embedding
        cross, _ = self.cross_attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )
        query = query + cross
        self_output, _ = self.self_attention(
            self.self_norm(query),
            self.self_norm(query),
            self.self_norm(query),
            need_weights=False,
        )
        query = query + self_output
        query = query + self.ffn(self.ffn_norm(query))
        return self.output_projection(query)


class TargetCameraActionEncoder(nn.Module):
    """Per-latent-frame camera action embedding for the LingBot timestep path."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, camera_vectors: torch.Tensor) -> torch.Tensor:
        return self.net(camera_vectors)
