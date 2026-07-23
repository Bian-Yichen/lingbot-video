from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera import RayFourierEncoder, plucker_rays, view_pose_features


class MemorySource(IntEnum):
    """Provenance controls how readily evidence may enter shared memory."""

    CAPTURE = 0
    GENERATED = 1
    VERIFIED_GENERATED = 2


@dataclass
class RecurrentSceneMemoryState:
    """Fixed-budget scene state.

    ``slow`` is shared, persistent scene knowledge. ``fast`` is branch-local
    episodic evidence and is the immediate destination for generated frames.
    """

    slow_tokens: torch.Tensor
    fast_tokens: torch.Tensor
    slow_confidence: torch.Tensor
    fast_confidence: torch.Tensor
    steps: torch.Tensor

    def detach(self) -> "RecurrentSceneMemoryState":
        return RecurrentSceneMemoryState(
            slow_tokens=self.slow_tokens.detach(),
            fast_tokens=self.fast_tokens.detach(),
            slow_confidence=self.slow_confidence.detach(),
            fast_confidence=self.fast_confidence.detach(),
            steps=self.steps.detach(),
        )

    def clone(self) -> "RecurrentSceneMemoryState":
        return RecurrentSceneMemoryState(
            slow_tokens=self.slow_tokens.clone(),
            fast_tokens=self.fast_tokens.clone(),
            slow_confidence=self.slow_confidence.clone(),
            fast_confidence=self.fast_confidence.clone(),
            steps=self.steps.clone(),
        )


@dataclass
class MemoryUpdateDiagnostics:
    fast_write_gate: torch.Tensor
    slow_write_gate: Optional[torch.Tensor]
    evidence_confidence: torch.Tensor
    source_type: torch.Tensor


@dataclass
class MemoryUpdateOutput:
    state: RecurrentSceneMemoryState
    diagnostics: MemoryUpdateDiagnostics


@dataclass
class GeometryQueryOutput:
    log_depth: torch.Tensor
    features: Optional[torch.Tensor]
    query_indices: torch.Tensor
    total_queries: int
    query_grid: tuple[int, int, int]


def _safe_padding_mask(mask: torch.Tensor) -> torch.Tensor:
    """MultiheadAttention cannot consume a row whose every key is masked."""

    safe = mask.clone()
    all_masked = safe.all(dim=-1)
    if all_masked.any():
        safe[all_masked, 0] = False
    return safe


class PerceiverMemoryBlock(nn.Module):
    """Cross-attend fixed state slots to a new observation, then mix slots."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.cross_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * width, width),
        )

    def forward(
        self,
        queries: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized_queries = self.query_norm(queries)
        normalized_context = self.context_norm(context)
        cross, _ = self.cross_attention(
            normalized_queries,
            normalized_context,
            normalized_context,
            key_padding_mask=(
                _safe_padding_mask(context_padding_mask)
                if context_padding_mask is not None
                else None
            ),
            need_weights=False,
        )
        queries = queries + cross
        normalized_queries = self.self_norm(queries)
        self_output, _ = self.self_attention(
            normalized_queries,
            normalized_queries,
            normalized_queries,
            need_weights=False,
        )
        queries = queries + self_output
        return queries + self.ffn(self.ffn_norm(queries))


class ObservationTokenizer(nn.Module):
    """Turn every posed latent frame into ray-addressed evidence tokens."""

    def __init__(
        self,
        latent_channels: int,
        width: int,
        input_grid: tuple[int, int],
        ray_fourier_bands: int,
    ):
        super().__init__()
        self.width = width
        self.input_grid = input_grid
        self.visual_projection = nn.Conv2d(latent_channels, width, kernel_size=1)
        self.ray_encoder = RayFourierEncoder(width, ray_fourier_bands)
        self.view_pose_projection = nn.Sequential(
            nn.Linear(13, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.source_embedding = nn.Embedding(len(MemorySource), width)
        self.confidence_projection = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        latents: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        valid_mask: torch.Tensor,
        source_type: torch.Tensor,
        confidence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim != 5:
            raise ValueError("latents must have shape (B, N, C, H, W)")
        batch, frames, channels, height, width = latents.shape
        if c2w.shape != (batch, frames, 4, 4):
            raise ValueError("c2w must have shape (B, N, 4, 4)")
        if intrinsics.shape != (batch, frames, 3, 3):
            raise ValueError("intrinsics must have shape (B, N, 3, 3)")
        if valid_mask.shape != (batch, frames):
            raise ValueError("valid_mask must have shape (B, N)")
        if source_type.shape != (batch, frames):
            raise ValueError("source_type must have shape (B, N)")
        if confidence.shape != (batch, frames):
            raise ValueError("confidence must have shape (B, N)")

        grid_height, grid_width = self.input_grid
        visual = self.visual_projection(
            latents.reshape(batch * frames, channels, height, width)
        )
        visual = F.adaptive_avg_pool2d(visual, self.input_grid)
        visual = visual.flatten(2).transpose(1, 2)

        rays = plucker_rays(c2w, intrinsics, grid_height, grid_width)
        rays = self.ray_encoder(rays).reshape(
            batch * frames, grid_height * grid_width, self.width
        )
        pose = self.view_pose_projection(
            view_pose_features(c2w, intrinsics).float()
        )
        provenance = self.source_embedding(source_type.long())
        confidence_embedding = self.confidence_projection(
            confidence.float().clamp(0.0, 1.0)[..., None]
        )
        frame_embedding = pose + provenance + confidence_embedding
        tokens = visual + rays.to(visual.dtype)
        tokens = tokens + frame_embedding.reshape(
            batch * frames, 1, self.width
        ).to(tokens.dtype)
        tokens = self.output_norm(tokens).reshape(
            batch, frames * grid_height * grid_width, self.width
        )
        padding_mask = ~valid_mask[:, :, None].expand(
            batch, frames, grid_height * grid_width
        ).reshape(batch, -1)
        return tokens, padding_mask


class GatedMemoryUpdater(nn.Module):
    """A confidence-aware delta-rule writer over fixed memory slots."""

    def __init__(self, width: int, heads: int, layers: int, confidence_decay: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [PerceiverMemoryBlock(width, heads) for _ in range(layers)]
        )
        self.write_gate = nn.Sequential(
            nn.Linear(2 * width + 3, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )
        self.output_norm = nn.LayerNorm(width)
        self.confidence_decay = confidence_decay

    def forward(
        self,
        memory: torch.Tensor,
        memory_confidence: torch.Tensor,
        context: torch.Tensor,
        context_padding_mask: torch.Tensor,
        evidence_confidence: torch.Tensor,
        write_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proposal = memory
        for block in self.blocks:
            proposal = block(
                proposal,
                context,
                context_padding_mask=context_padding_mask,
            )
        proposal = self.output_norm(proposal)
        novelty = (proposal.float() - memory.float()).square().mean(
            dim=-1, keepdim=True
        ).sqrt()
        batch, slots = memory.shape[:2]
        evidence = evidence_confidence[:, None, None].expand(batch, slots, 1)
        old_confidence = memory_confidence[..., None].float()
        gate_input = torch.cat(
            (
                memory.float(),
                proposal.float(),
                novelty,
                old_confidence,
                evidence.float(),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.write_gate(gate_input))
        gate = gate * evidence * write_scale[:, None, None].float()
        updated = memory.float() + gate * (proposal.float() - memory.float())
        updated = self.output_norm(updated).to(memory.dtype)
        candidate_confidence = torch.maximum(
            memory_confidence.float() * self.confidence_decay,
            gate.squeeze(-1) * evidence_confidence[:, None].float(),
        ).clamp(0.0, 1.0)
        active = (write_scale > 0)[:, None, None]
        updated = torch.where(active, updated, memory)
        updated_confidence = torch.where(
            active.squeeze(-1),
            candidate_confidence,
            memory_confidence.float(),
        )
        return updated, updated_confidence.to(memory.dtype), gate.squeeze(-1)


class PersistentRecurrentSceneMemory(nn.Module):
    """Pose-free hierarchical state with constant storage and online writes.

    All capture frames are consumed in order. Generated frames enter ``fast``
    immediately, so the very next rollout can use them. Consolidation into
    ``slow`` is provenance gated: observed and verified evidence writes
    normally, while unverified generated evidence is deliberately conservative.
    """

    def __init__(
        self,
        *,
        latent_channels: int,
        condition_dim: int,
        width: int,
        heads: int,
        slow_tokens: int,
        fast_tokens: int,
        update_layers: int,
        consolidation_layers: int,
        input_grid: tuple[int, int],
        ray_fourier_bands: int,
        confidence_decay: float,
        generated_slow_write_scale: float,
        verified_slow_write_scale: float,
    ):
        super().__init__()
        self.width = width
        self.slow_token_count = slow_tokens
        self.fast_token_count = fast_tokens
        self.generated_slow_write_scale = generated_slow_write_scale
        self.verified_slow_write_scale = verified_slow_write_scale
        self.initial_slow = nn.Parameter(
            torch.randn(1, slow_tokens, width) / width**0.5
        )
        self.initial_fast = nn.Parameter(
            torch.randn(1, fast_tokens, width) / width**0.5
        )
        self.tokenizer = ObservationTokenizer(
            latent_channels,
            width,
            input_grid,
            ray_fourier_bands,
        )
        self.fast_writer = GatedMemoryUpdater(
            width, heads, update_layers, confidence_decay
        )
        self.slow_writer = GatedMemoryUpdater(
            width, heads, consolidation_layers, confidence_decay
        )
        self.confidence_projection = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.output_norm = nn.LayerNorm(width)
        self.condition_projection = nn.Linear(width, condition_dim)
        self.condition_gate = nn.Parameter(torch.zeros(()))

    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RecurrentSceneMemoryState:
        return RecurrentSceneMemoryState(
            slow_tokens=self.initial_slow.expand(batch_size, -1, -1).to(
                device=device, dtype=dtype
            ),
            fast_tokens=self.initial_fast.expand(batch_size, -1, -1).to(
                device=device, dtype=dtype
            ),
            slow_confidence=torch.zeros(
                batch_size,
                self.slow_token_count,
                device=device,
                dtype=dtype,
            ),
            fast_confidence=torch.zeros(
                batch_size,
                self.fast_token_count,
                device=device,
                dtype=dtype,
            ),
            steps=torch.zeros(batch_size, device=device, dtype=torch.long),
        )

    @staticmethod
    def _expand_source_type(
        source_type: int | MemorySource | torch.Tensor,
        batch: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(source_type, torch.Tensor):
            expanded = source_type.to(device=device, dtype=torch.long)
            if expanded.ndim == 0:
                expanded = expanded.expand(batch, frames)
            elif expanded.ndim == 1:
                if expanded.shape[0] != batch:
                    raise ValueError("1D source_type must have shape (B,)")
                expanded = expanded[:, None].expand(batch, frames)
            elif expanded.shape != (batch, frames):
                raise ValueError("source_type must be scalar, (B,), or (B,N)")
            return expanded
        return torch.full(
            (batch, frames),
            int(source_type),
            device=device,
            dtype=torch.long,
        )

    @staticmethod
    def _evidence_confidence(
        confidence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = valid_mask.to(confidence.dtype)
        return (
            (confidence * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        ).clamp(0.0, 1.0)

    def _slow_write_scale(
        self,
        source_type: torch.Tensor,
        valid_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        observed = (source_type == int(MemorySource.CAPTURE)).to(dtype)
        generated = (source_type == int(MemorySource.GENERATED)).to(dtype)
        verified = (
            source_type == int(MemorySource.VERIFIED_GENERATED)
        ).to(dtype)
        per_frame = (
            observed
            + generated * self.generated_slow_write_scale
            + verified * self.verified_slow_write_scale
        )
        weights = valid_mask.to(dtype)
        return (
            (per_frame * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        ).clamp(0.0, 1.0)

    def update(
        self,
        state: RecurrentSceneMemoryState,
        *,
        latents: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        valid_mask: torch.Tensor,
        source_type: int | MemorySource | torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        consolidate: bool = False,
    ) -> MemoryUpdateOutput:
        batch, frames = latents.shape[:2]
        valid_mask = valid_mask.bool()
        source = self._expand_source_type(
            source_type, batch, frames, latents.device
        )
        if confidence is None:
            confidence = latents.new_ones(batch, frames)
        if confidence.shape != (batch, frames):
            raise ValueError("confidence must have shape (B, N)")
        confidence = confidence.to(latents.dtype).clamp(0.0, 1.0)
        context, context_padding_mask = self.tokenizer(
            latents,
            c2w,
            intrinsics,
            valid_mask,
            source,
            confidence,
        )
        evidence_confidence = self._evidence_confidence(
            confidence, valid_mask
        )
        has_evidence = valid_mask.any(dim=1).to(latents.dtype)
        fast_tokens, fast_confidence, fast_gate = self.fast_writer(
            state.fast_tokens,
            state.fast_confidence,
            context,
            context_padding_mask,
            evidence_confidence,
            has_evidence,
        )
        next_state = RecurrentSceneMemoryState(
            slow_tokens=state.slow_tokens,
            fast_tokens=fast_tokens,
            slow_confidence=state.slow_confidence,
            fast_confidence=fast_confidence,
            steps=state.steps + valid_mask.long().sum(dim=1),
        )
        slow_gate = None
        if consolidate:
            has_fast_evidence = next_state.fast_confidence.bool().any(
                dim=1, keepdim=True
            )
            next_state, slow_gate = self.consolidate(
                next_state,
                source_type=source[:, :1],
                source_valid_mask=has_fast_evidence,
            )
        return MemoryUpdateOutput(
            state=next_state,
            diagnostics=MemoryUpdateDiagnostics(
                fast_write_gate=fast_gate,
                slow_write_gate=slow_gate,
                evidence_confidence=evidence_confidence,
                source_type=source,
            ),
        )

    def consolidate(
        self,
        state: RecurrentSceneMemoryState,
        *,
        source_type: torch.Tensor,
        source_valid_mask: torch.Tensor,
    ) -> tuple[RecurrentSceneMemoryState, torch.Tensor]:
        batch = state.slow_tokens.shape[0]
        fast_padding_mask = state.fast_confidence <= 0
        evidence_confidence = state.fast_confidence.float().mean(dim=1)
        write_scale = self._slow_write_scale(
            source_type,
            source_valid_mask,
            state.slow_tokens.dtype,
        )
        slow_tokens, slow_confidence, slow_gate = self.slow_writer(
            state.slow_tokens,
            state.slow_confidence,
            state.fast_tokens,
            fast_padding_mask,
            evidence_confidence,
            write_scale,
        )
        reset_fast = self.initial_fast.expand(batch, -1, -1).to(
            state.fast_tokens
        )
        return (
            RecurrentSceneMemoryState(
                slow_tokens=slow_tokens,
                fast_tokens=reset_fast,
                slow_confidence=slow_confidence,
                fast_confidence=torch.zeros_like(state.fast_confidence),
                steps=state.steps,
            ),
            slow_gate,
        )

    def encode_stream(
        self,
        *,
        latents: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        valid_mask: torch.Tensor,
        chunk_size: int,
        consolidation_interval: int,
        truncate_bptt_every: int = 0,
        initial_state: Optional[RecurrentSceneMemoryState] = None,
    ) -> RecurrentSceneMemoryState:
        if latents.ndim != 5:
            raise ValueError("capture latents must have shape (B, N, C, H, W)")
        if not valid_mask.bool().any(dim=1).all():
            raise ValueError("Every scene must contain at least one valid capture")
        batch, frames = latents.shape[:2]
        state = (
            self.initial_state(batch, latents.device, latents.dtype)
            if initial_state is None
            else initial_state
        )
        chunks_since_consolidation = 0
        chunk_index = 0
        last_source = None
        last_valid = None
        for start in range(0, frames, chunk_size):
            if (
                self.training
                and truncate_bptt_every > 0
                and chunk_index > 0
                and chunk_index % truncate_bptt_every == 0
            ):
                state = state.detach()
            end = min(start + chunk_size, frames)
            chunk_valid = valid_mask[:, start:end].bool()
            chunk_source = torch.full(
                chunk_valid.shape,
                int(MemorySource.CAPTURE),
                device=latents.device,
                dtype=torch.long,
            )
            chunks_since_consolidation += 1
            should_consolidate = (
                chunks_since_consolidation >= consolidation_interval
            )
            output = self.update(
                state,
                latents=latents[:, start:end],
                c2w=c2w[:, start:end],
                intrinsics=intrinsics[:, start:end],
                valid_mask=chunk_valid,
                source_type=chunk_source,
                consolidate=should_consolidate,
            )
            state = output.state
            last_source = chunk_source
            last_valid = chunk_valid
            if should_consolidate:
                chunks_since_consolidation = 0
            chunk_index += 1
        if chunks_since_consolidation > 0:
            if last_source is None or last_valid is None:
                raise RuntimeError("The capture stream unexpectedly contained no chunks")
            has_fast_evidence = state.fast_confidence.bool().any(
                dim=1, keepdim=True
            )
            state, _ = self.consolidate(
                state,
                source_type=last_source[:, :1],
                source_valid_mask=has_fast_evidence,
            )
        return state

    def scene_tokens(self, state: RecurrentSceneMemoryState) -> torch.Tensor:
        tokens = torch.cat((state.slow_tokens, state.fast_tokens), dim=1)
        confidence = torch.cat(
            (state.slow_confidence, state.fast_confidence), dim=1
        )
        tokens = tokens + self.confidence_projection(
            confidence.float()[..., None]
        ).to(tokens.dtype)
        return self.output_norm(tokens)

    def condition_tokens(
        self, state: RecurrentSceneMemoryState
    ) -> torch.Tensor:
        return self.condition_gate.tanh() * self.condition_projection(
            self.scene_tokens(state)
        )


class RayQueryableGeometryHead(nn.Module):
    """Training head that asks the implicit memory ray-conditioned questions."""

    def __init__(
        self,
        memory_width: int,
        heads: int,
        ray_fourier_bands: int,
        feature_dim: int = 0,
    ):
        super().__init__()
        self.ray_encoder = RayFourierEncoder(memory_width, ray_fourier_bands)
        self.query_norm = nn.LayerNorm(memory_width)
        self.memory_norm = nn.LayerNorm(memory_width)
        self.cross_attention = nn.MultiheadAttention(
            memory_width, heads, batch_first=True
        )
        self.self_attention = nn.MultiheadAttention(
            memory_width, heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(memory_width),
            nn.Linear(memory_width, 4 * memory_width),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * memory_width, memory_width),
        )
        self.depth_head = nn.Sequential(
            nn.LayerNorm(memory_width),
            nn.Linear(memory_width, 1),
        )
        self.feature_head = (
            nn.Sequential(
                nn.LayerNorm(memory_width),
                nn.Linear(memory_width, feature_dim),
            )
            if feature_dim > 0
            else None
        )

    def forward(
        self,
        scene_tokens: torch.Tensor,
        query_c2w: torch.Tensor,
        query_intrinsics: torch.Tensor,
        grid_height: int,
        grid_width: int,
        max_queries: int,
        query_indices: Optional[torch.Tensor] = None,
    ) -> GeometryQueryOutput:
        rays = plucker_rays(
            query_c2w, query_intrinsics, grid_height, grid_width
        )
        batch, frames = rays.shape[:2]
        flat_rays = rays.reshape(batch, -1, rays.shape[-1])
        total_queries = flat_rays.shape[1]
        if query_indices is None:
            count = min(max_queries, total_queries)
            if self.training and count < total_queries:
                query_indices = torch.rand(
                    batch, total_queries, device=flat_rays.device
                ).topk(count, dim=-1).indices
            else:
                query_indices = torch.linspace(
                    0,
                    total_queries - 1,
                    count,
                    device=flat_rays.device,
                ).long()[None].expand(batch, -1)
        gather_index = query_indices[..., None].expand(
            batch, query_indices.shape[1], flat_rays.shape[-1]
        )
        selected_rays = torch.gather(flat_rays, dim=1, index=gather_index)
        queries = self.ray_encoder(selected_rays)
        normalized_memory = self.memory_norm(scene_tokens)
        cross, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_memory,
            normalized_memory,
            need_weights=False,
        )
        queries = queries + cross
        normalized = self.query_norm(queries)
        self_output, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        queries = queries + self_output
        queries = queries + self.ffn(queries)
        log_depth = self.depth_head(queries).squeeze(-1)
        features = (
            self.feature_head(queries) if self.feature_head is not None else None
        )
        return GeometryQueryOutput(
            log_depth=log_depth,
            features=features,
            query_indices=query_indices,
            total_queries=total_queries,
            query_grid=(frames, grid_height, grid_width),
        )
