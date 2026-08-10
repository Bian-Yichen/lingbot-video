from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import ActiveMemoryRollout, ActiveWorldMemoryAgent, HierarchicalMemoryIndex
from .geometry import camera_descriptor


@dataclass(frozen=True)
class ActiveWorldMemoryConfig:
    memory_dim: int = 512
    num_heads: int = 8
    coarse_grid_h: int = 4
    coarse_grid_w: int = 7
    canvas_grid_h: int = 4
    canvas_grid_w: int = 7
    condition_tokens: int = 64
    latent_channels: int = 16
    text_dim: int = 2560
    max_retrieval_steps: int = 24
    min_selected_views: int = 2
    max_selected_views: int = 24
    retrieval_cost: float = 0.01
    coarse_encoder_chunk: int = 8
    max_generated_views: int = 256

    def validate(self) -> None:
        if self.memory_dim % self.num_heads:
            raise ValueError("memory_dim must be divisible by num_heads")
        if self.memory_dim % 32:
            raise ValueError("memory_dim must be divisible by 32 for GroupNorm")
        if min(
            self.coarse_grid_h,
            self.coarse_grid_w,
            self.canvas_grid_h,
            self.canvas_grid_w,
            self.condition_tokens,
        ) < 1:
            raise ValueError("all token-grid sizes must be positive")
        if self.max_retrieval_steps < self.min_selected_views:
            raise ValueError("max_retrieval_steps cannot be smaller than min_selected_views")
        if self.max_selected_views > self.max_retrieval_steps:
            raise ValueError(
                "one view is retrieved per step, so max_selected_views cannot "
                "exceed max_retrieval_steps"
            )
        if self.max_generated_views < 1:
            raise ValueError("max_generated_views must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FourierCameraEncoder(nn.Module):
    def __init__(self, dim: int, frequencies: int = 4) -> None:
        super().__init__()
        self.frequencies = frequencies
        descriptor_dim = 14
        input_dim = descriptor_dim * (1 + 2 * frequencies)
        self.network = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        image_hw: tuple[int, int],
        times: torch.Tensor,
    ) -> torch.Tensor:
        descriptor = camera_descriptor(c2w, intrinsics, image_hw, times)
        features = [descriptor]
        for exponent in range(self.frequencies):
            frequency = (2.0**exponent) * math.pi
            features.extend((torch.sin(frequency * descriptor), torch.cos(frequency * descriptor)))
        return self.network(torch.cat(features, dim=-1))


class CoarseVisualMemoryEncoder(nn.Module):
    """Encode all candidate views cheaply; expensive VAE runs only after STOP."""

    def __init__(self, dim: int, grid_hw: tuple[int, int], chunk_size: int) -> None:
        super().__init__()
        self.grid_hw = grid_hw
        self.chunk_size = chunk_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.Conv2d(128, dim, 3, stride=2, padding=1),
            nn.GroupNorm(32, dim),
            nn.SiLU(),
        )
        self.patch_norm = nn.LayerNorm(dim)
        self.view_projection = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if rgb.ndim != 5 or rgb.shape[2] != 3:
            raise ValueError(f"candidate RGB must be (B,N,3,H,W), got {tuple(rgb.shape)}")
        batch, views = rgb.shape[:2]
        flat = rgb.reshape(batch * views, *rgb.shape[2:])
        device = next(self.parameters()).device
        chunks: list[torch.Tensor] = []
        for start in range(0, flat.shape[0], self.chunk_size):
            value = flat[start : start + self.chunk_size].to(
                device=device, non_blocking=True
            )
            if value.dtype == torch.uint8:
                value = value.float().div_(255.0)
            else:
                value = value.float().clamp_(0.0, 1.0)
            # The index encoder never needs generation resolution.
            value = F.interpolate(value, size=(128, 224), mode="bilinear", align_corners=False)
            value = self.encoder(value)
            value = F.adaptive_avg_pool2d(value, self.grid_hw)
            chunks.append(value)
        feature = torch.cat(chunks, dim=0)
        patches = feature.flatten(2).transpose(1, 2)
        patches = self.patch_norm(patches).reshape(batch, views, -1, feature.shape[1])
        view_tokens = self.view_projection(patches.mean(dim=2))
        return view_tokens, patches


class FineLatentEvidenceEncoder(nn.Module):
    """Preserve appearance details from independently encoded selected views."""

    def __init__(self, latent_channels: int, dim: int, grid_hw: tuple[int, int]) -> None:
        super().__init__()
        self.grid_hw = grid_hw
        self.projection = nn.Sequential(
            nn.Conv2d(latent_channels, dim, 2, stride=2),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        latents: torch.Tensor,
        camera_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latents.ndim != 5:
            raise ValueError("selected view latents must be (B,N,C,H,W)")
        batch, views, channels, height, width = latents.shape
        feature = self.projection(latents.reshape(batch * views, channels, height, width))
        feature = F.adaptive_avg_pool2d(feature, self.grid_hw)
        patches = feature.flatten(2).transpose(1, 2).reshape(batch, views, -1, feature.shape[1])
        patches = self.norm(patches + camera_tokens.unsqueeze(2))
        return patches.mean(dim=2), patches


class TargetCanvasEncoder(nn.Module):
    def __init__(self, dim: int, grid_hw: tuple[int, int]) -> None:
        super().__init__()
        self.grid_hw = grid_hw
        regions = grid_hw[0] * grid_hw[1]
        self.region_embeddings = nn.Parameter(torch.randn(regions, dim) * 0.02)
        self.fusion = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim), nn.LayerNorm(dim)
        )

    def forward(self, camera_tokens: torch.Tensor) -> torch.Tensor:
        batch, frames, dim = camera_tokens.shape
        regions = self.region_embeddings.view(1, 1, -1, dim).expand(batch, frames, -1, -1)
        cameras = camera_tokens.unsqueeze(2).expand_as(regions)
        return self.fusion(torch.cat((cameras, regions), dim=-1)).reshape(batch, -1, dim)


class WorldFeatureCritic(nn.Module):
    """Cheap target-latent predictor used for information-gain supervision."""

    def __init__(self, dim: int, latent_channels: int) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, latent_channels)
        )

    def forward(self, canvas: torch.Tensor) -> torch.Tensor:
        return self.predictor(canvas)


class MemoryConditionResampler(nn.Module):
    def __init__(self, dim: int, text_dim: int, num_heads: int, num_tokens: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.to_text = nn.Sequential(nn.Linear(dim, text_dim), nn.LayerNorm(text_dim))

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        queries = self.queries.unsqueeze(0).expand(evidence.shape[0], -1, -1)
        output, _ = self.attention(queries, self.norm(evidence), evidence, need_weights=False)
        return self.to_text(output)


def _pool_episodes(
    view_tokens: torch.Tensor,
    episode_ids: torch.Tensor,
    view_mask: torch.Tensor,
    view_confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, views, dim = view_tokens.shape
    max_episodes = int(episode_ids.max().item()) + 1
    output = view_tokens.new_zeros(batch, max_episodes, dim)
    counts = view_tokens.new_zeros(batch, max_episodes, 1)
    for batch_index in range(batch):
        valid_ids = episode_ids[batch_index].masked_fill(~view_mask[batch_index], 0)
        # scatter_add_ requires its source to have exactly the destination
        # dtype.  Confidence may arrive as fp32 while view tokens are produced
        # under mixed precision, so normalize it explicitly here.
        weights = (
            view_mask[batch_index].to(view_tokens.dtype)
            * view_confidence[batch_index].to(view_tokens.dtype)
        )
        weighted = view_tokens[batch_index] * weights.unsqueeze(-1)
        output[batch_index].scatter_add_(
            0, valid_ids.unsqueeze(-1).expand(-1, dim), weighted
        )
        counts[batch_index].scatter_add_(
            0, valid_ids.unsqueeze(-1), weights.unsqueeze(-1)
        )
    mask = counts.squeeze(-1) > 0
    return output / counts.clamp_min(1.0), mask


class ActiveWorldMemoryModel(nn.Module):
    """Active visual recall around an unchanged LingBot-Video DiT backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        config: ActiveWorldMemoryConfig = ActiveWorldMemoryConfig(),
    ) -> None:
        super().__init__()
        config.validate()
        backbone_config = getattr(backbone, "config", None)
        backbone_text_dim = getattr(backbone_config, "text_dim", config.text_dim)
        backbone_channels = getattr(
            backbone_config, "in_channels", config.latent_channels
        )
        if int(backbone_text_dim) != config.text_dim:
            raise ValueError(
                f"memory text_dim={config.text_dim} does not match backbone "
                f"text_dim={backbone_text_dim}"
            )
        if int(backbone_channels) != config.latent_channels:
            raise ValueError(
                f"memory latent_channels={config.latent_channels} does not match "
                f"backbone in_channels={backbone_channels}"
            )
        self.backbone = backbone
        self.active_config = config
        dim = config.memory_dim
        coarse_grid = (config.coarse_grid_h, config.coarse_grid_w)
        canvas_grid = (config.canvas_grid_h, config.canvas_grid_w)
        self.camera_encoder = FourierCameraEncoder(dim)
        self.coarse_encoder = CoarseVisualMemoryEncoder(
            dim, coarse_grid, config.coarse_encoder_chunk
        )
        self.fine_encoder = FineLatentEvidenceEncoder(
            config.latent_channels, dim, coarse_grid
        )
        self.target_canvas_encoder = TargetCanvasEncoder(dim, canvas_grid)
        self.agent = ActiveWorldMemoryAgent(
            dim=dim,
            num_heads=config.num_heads,
            max_steps=config.max_retrieval_steps,
            min_selected_views=config.min_selected_views,
            max_selected_views=config.max_selected_views,
            retrieval_cost=config.retrieval_cost,
        )
        self.world_critic = WorldFeatureCritic(dim, config.latent_channels)
        self.condition_resampler = MemoryConditionResampler(
            dim, config.text_dim, config.num_heads, config.condition_tokens
        )
        self.camera_to_text = nn.Sequential(
            nn.Linear(dim, config.text_dim), nn.LayerNorm(config.text_dim)
        )
        self.condition_type = nn.Parameter(torch.randn(1, 1, config.text_dim) * 0.02)
        self.memory_type_embeddings = nn.Parameter(torch.randn(2, dim) * 0.02)

    def forward(self, operation: str = "generator", **kwargs):
        """DDP-safe dispatch: one outer forward owns a complete training step."""

        if operation == "generator":
            return self.generator_forward(**kwargs)
        if operation == "training_step":
            # Local import avoids a module cycle while ensuring Accelerate/DDP
            # observes one forward that touches every active submodule.
            from .training import _active_world_training_step_impl

            return _active_world_training_step_impl(self, **kwargs)
        raise ValueError(f"unsupported ActiveWorldMemoryModel operation {operation!r}")

    @property
    def canvas_regions(self) -> int:
        return self.active_config.canvas_grid_h * self.active_config.canvas_grid_w

    @staticmethod
    def _to_float_rgb(rgb: torch.Tensor, device: torch.device) -> torch.Tensor:
        rgb = rgb.to(device=device, non_blocking=True)
        if rgb.dtype == torch.uint8:
            return rgb.float().div_(255.0)
        return rgb.float().clamp_(0.0, 1.0)

    @staticmethod
    def latent_pose_indices(rgb_frames: int, latent_frames: int, device: torch.device) -> torch.Tensor:
        if latent_frames == 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        return torch.linspace(0, rgb_frames - 1, latent_frames, device=device).round().long()

    def encode_cameras(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        image_hw: tuple[int, int],
        times: torch.Tensor,
    ) -> torch.Tensor:
        return self.camera_encoder(c2w, intrinsics, image_hw, times)

    def build_memory_index(
        self,
        candidate_rgb: torch.Tensor,
        candidate_c2w: torch.Tensor,
        candidate_intrinsics: torch.Tensor,
        candidate_times: torch.Tensor,
        episode_ids: torch.Tensor,
        image_hw: tuple[int, int],
        view_mask: torch.Tensor | None = None,
    ) -> HierarchicalMemoryIndex:
        device = next(self.coarse_encoder.parameters()).device
        candidate_c2w = candidate_c2w.to(device)
        candidate_intrinsics = candidate_intrinsics.to(device)
        candidate_times = candidate_times.to(device)
        episode_ids = episode_ids.to(device)
        if view_mask is None:
            view_mask = torch.ones(
                candidate_rgb.shape[:2], dtype=torch.bool, device=device
            )
        camera = self.encode_cameras(
            candidate_c2w, candidate_intrinsics, image_hw, candidate_times
        )
        view_tokens, patch_tokens = self.coarse_encoder(candidate_rgb)
        capture_type = self.memory_type_embeddings[0]
        view_tokens = view_tokens + camera + capture_type
        patch_tokens = patch_tokens + camera.unsqueeze(2) + capture_type
        view_confidence = torch.ones(
            view_tokens.shape[:2], device=device, dtype=torch.float32
        )
        view_provenance = torch.zeros(
            view_tokens.shape[:2], device=device, dtype=torch.long
        )
        episode_tokens, episode_mask = _pool_episodes(
            view_tokens, episode_ids, view_mask, view_confidence
        )
        return HierarchicalMemoryIndex(
            view_tokens=view_tokens,
            patch_tokens=patch_tokens,
            episode_tokens=episode_tokens,
            episode_mask=episode_mask,
            view_episode_ids=episode_ids,
            view_mask=view_mask,
            view_confidence=view_confidence,
            view_provenance=view_provenance,
        )

    def target_canvas(
        self,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
        target_times: torch.Tensor,
        image_hw: tuple[int, int],
        latent_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        indices = self.latent_pose_indices(
            target_c2w.shape[1], latent_frames, target_c2w.device
        )
        camera_tokens = self.encode_cameras(
            target_c2w[:, indices],
            target_intrinsics[:, indices],
            image_hw,
            target_times[:, indices],
        )
        return self.target_canvas_encoder(camera_tokens), camera_tokens

    def target_features(self, target_latents: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, _, _ = target_latents.shape
        pooled = F.adaptive_avg_pool3d(
            target_latents.float(),
            (
                frames,
                self.active_config.canvas_grid_h,
                self.active_config.canvas_grid_w,
            ),
        )
        return pooled.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels).detach()

    def encode_fine_latents(
        self,
        selected_latents: torch.Tensor,
        selected_camera_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fine_encoder(selected_latents, selected_camera_tokens)

    def make_condition_tokens(
        self,
        rollout: ActiveMemoryRollout,
        memory: HierarchicalMemoryIndex,
        target_camera_tokens: torch.Tensor,
        fine_view_tokens: torch.Tensor | None = None,
        fine_patch_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        evidence = [rollout.state.canvas]
        if rollout.selected_views:
            indices = torch.tensor(
                rollout.selected_views,
                device=memory.view_tokens.device,
                dtype=torch.long,
            )
            evidence.append(memory.view_tokens.index_select(1, indices))
        if fine_view_tokens is not None:
            evidence.append(fine_view_tokens)
        if fine_patch_tokens is not None:
            evidence.append(fine_patch_tokens.flatten(1, 2))
        resampled = self.condition_resampler(torch.cat(evidence, dim=1))
        camera = self.camera_to_text(target_camera_tokens)
        return torch.cat((camera, resampled), dim=1) + self.condition_type

    @staticmethod
    def append_prompt_condition(
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        condition_tokens: torch.Tensor,
        *,
        drop_condition: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if drop_condition:
            return prompt_embeds, prompt_mask
        condition_mask = torch.ones(
            condition_tokens.shape[:2],
            device=prompt_mask.device,
            dtype=prompt_mask.dtype,
        )
        return (
            torch.cat((prompt_embeds, condition_tokens.to(prompt_embeds.dtype)), dim=1),
            torch.cat((prompt_mask, condition_mask), dim=1),
        )

    def generator_forward(
        self,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        condition_tokens: torch.Tensor,
        *,
        drop_condition: bool = False,
    ) -> torch.Tensor:
        encoder_hidden_states, attention_mask = self.append_prompt_condition(
            prompt_embeds,
            prompt_mask,
            condition_tokens,
            drop_condition=drop_condition,
        )
        return self.backbone(
            noisy_latents,
            timestep,
            encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            return_dict=False,
        )[0]

    def append_latent_memory(
        self,
        memory: HierarchicalMemoryIndex,
        latent_views: torch.Tensor,
        camera_tokens: torch.Tensor,
        confidence: float | torch.Tensor = 0.5,
    ) -> HierarchicalMemoryIndex:
        """Dynamically add generated/teacher-forced views for later chunks."""

        if memory.view_tokens.shape[0] != 1:
            raise ValueError(
                "dynamic memory update currently requires one scene per rank"
            )
        view_tokens, patch_tokens = self.fine_encoder(latent_views, camera_tokens)
        generated_type = self.memory_type_embeddings[1]
        view_tokens = view_tokens + generated_type
        patch_tokens = patch_tokens + generated_type
        view_episode = torch.full(
            (memory.view_episode_ids.shape[0], view_tokens.shape[1]),
            int(memory.view_episode_ids.max().item()) + 1,
            device=memory.view_episode_ids.device,
            dtype=memory.view_episode_ids.dtype,
        )
        view_mask = torch.cat(
            (
                memory.view_mask,
                torch.ones(
                    view_tokens.shape[:2],
                    device=memory.view_mask.device,
                    dtype=torch.bool,
                ),
            ),
            dim=1,
        )
        ids = torch.cat((memory.view_episode_ids, view_episode), dim=1)
        views = torch.cat((memory.view_tokens, view_tokens), dim=1)
        patches = torch.cat((memory.patch_tokens, patch_tokens), dim=1)
        new_confidence = torch.as_tensor(
            confidence, device=views.device, dtype=torch.float32
        )
        if new_confidence.ndim == 0:
            new_confidence = new_confidence.expand(view_tokens.shape[:2])
        elif tuple(new_confidence.shape) != tuple(view_tokens.shape[:2]):
            raise ValueError(
                "dynamic memory confidence must be scalar or have shape "
                f"{tuple(view_tokens.shape[:2])}"
            )
        confidences = torch.cat((memory.view_confidence, new_confidence), dim=1)
        provenance = torch.cat(
            (
                memory.view_provenance,
                torch.ones(
                    view_tokens.shape[:2], device=views.device, dtype=torch.long
                ),
            ),
            dim=1,
        )

        # Bound only self-generated memory. Capture observations remain intact;
        # half of the dynamic budget preserves recent context and half preserves
        # the highest-confidence older observations.
        generated_indices = torch.nonzero(
            provenance[0].eq(1), as_tuple=False
        ).flatten()
        if generated_indices.numel() > self.active_config.max_generated_views:
            budget = self.active_config.max_generated_views
            recent_count = max(1, budget // 2)
            recent = generated_indices[-recent_count:]
            older = generated_indices[:-recent_count]
            quality_count = budget - recent_count
            if quality_count > 0 and older.numel() > 0:
                quality = older[
                    confidences[0, older].topk(
                        min(quality_count, older.numel())
                    ).indices
                ]
                kept_generated = torch.cat((quality, recent)).unique(sorted=True)
            else:
                kept_generated = recent
            capture_indices = torch.nonzero(
                provenance[0].eq(0), as_tuple=False
            ).flatten()
            keep = torch.cat((capture_indices, kept_generated)).sort().values
            views = views.index_select(1, keep)
            patches = patches.index_select(1, keep)
            view_mask = view_mask.index_select(1, keep)
            ids = ids.index_select(1, keep)
            confidences = confidences.index_select(1, keep)
            provenance = provenance.index_select(1, keep)
            # Re-label sparse episode ids after pruning.
            _, ids = torch.unique(ids, sorted=True, return_inverse=True)
            ids = ids.reshape(1, -1)

        episodes, episode_mask = _pool_episodes(
            views, ids, view_mask, confidences
        )
        return HierarchicalMemoryIndex(
            view_tokens=views,
            patch_tokens=patches,
            episode_tokens=episodes,
            episode_mask=episode_mask,
            view_episode_ids=ids,
            view_mask=view_mask,
            view_confidence=confidences,
            view_provenance=provenance,
        )
