from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .camera import (
    CameraTokenEncoder,
    compute_scene_normalization,
    compute_world_to_scene_rotation,
    normalize_camera_poses,
)
from .config import LongSceneConfig
from .memory import (
    GeometryQueryOutput,
    MemorySource,
    MemoryUpdateDiagnostics,
    PersistentRecurrentSceneMemory,
    RayQueryableGeometryHead,
    RecurrentSceneMemoryState,
)


@dataclass
class LongSceneMemoryEncoding:
    state: RecurrentSceneMemoryState
    scene_tokens: torch.Tensor
    condition_tokens: torch.Tensor
    scene_center: torch.Tensor
    scene_scale: torch.Tensor
    world_to_scene_rotation: torch.Tensor


@dataclass
class LongSceneModelOutput:
    predicted_velocity: torch.Tensor
    memory: LongSceneMemoryEncoding
    paired_predicted_velocity: Optional[torch.Tensor] = None
    geometry: Optional[GeometryQueryOutput] = None
    memory_reconstruction: Optional[GeometryQueryOutput] = None
    updated_memory: Optional[LongSceneMemoryEncoding] = None
    teacher_updated_memory: Optional[LongSceneMemoryEncoding] = None
    rollout_predicted_velocity: Optional[torch.Tensor] = None
    memory_update_diagnostics: Optional[MemoryUpdateDiagnostics] = None


def _resolve_backbone_config(backbone: nn.Module):
    config = getattr(backbone, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return config
    get_base_model = getattr(backbone, "get_base_model", None)
    if get_base_model is not None:
        base = get_base_model()
        config = getattr(base, "config", None)
        if config is not None and hasattr(config, "hidden_size"):
            return config
    raise ValueError("Unable to resolve the LingBot backbone configuration")


class LongSceneWorldModel(nn.Module):
    """LingBot-Video with recurrent scene state and ray camera control."""

    def __init__(self, backbone: nn.Module, config: LongSceneConfig):
        super().__init__()
        self.backbone = backbone
        self.long_scene_config = config
        backbone_config = _resolve_backbone_config(backbone)
        self.backbone_hidden_size = int(backbone_config.hidden_size)
        self.text_dim = int(backbone_config.text_dim)
        self.latent_channels = int(backbone_config.in_channels)
        self.patch_size = tuple(backbone_config.patch_size)

        self.scene_memory = PersistentRecurrentSceneMemory(
            latent_channels=self.latent_channels,
            condition_dim=self.backbone_hidden_size,
            width=config.memory_width,
            heads=config.memory_heads,
            slow_tokens=config.slow_memory_tokens,
            fast_tokens=config.fast_memory_tokens,
            update_layers=config.memory_update_layers,
            consolidation_layers=config.memory_consolidation_layers,
            input_grid=(
                config.memory_input_grid_h,
                config.memory_input_grid_w,
            ),
            ray_fourier_bands=config.ray_fourier_bands,
            confidence_decay=config.memory_confidence_decay,
            generated_slow_write_scale=config.generated_slow_write_scale,
            verified_slow_write_scale=config.verified_slow_write_scale,
        )
        self.camera_encoder = CameraTokenEncoder(
            hidden_size=self.backbone_hidden_size,
            ray_fourier_bands=config.ray_fourier_bands,
        )
        self.geometry_head = RayQueryableGeometryHead(
            memory_width=config.memory_width,
            heads=config.memory_heads,
            ray_fourier_bands=config.ray_fourier_bands,
            feature_dim=config.geometry_feature_dim,
        )
        self.memory_readout_head = RayQueryableGeometryHead(
            memory_width=config.memory_width,
            heads=config.memory_heads,
            ray_fourier_bands=config.ray_fourier_bands,
            feature_dim=self.latent_channels,
        )
        self.memory_readout_head.depth_head.requires_grad_(False)

    def _wrap_memory(
        self,
        state: RecurrentSceneMemoryState,
        *,
        scene_center: torch.Tensor,
        scene_scale: torch.Tensor,
        world_to_scene_rotation: torch.Tensor,
    ) -> LongSceneMemoryEncoding:
        return LongSceneMemoryEncoding(
            state=state,
            scene_tokens=self.scene_memory.scene_tokens(state),
            condition_tokens=self.scene_memory.condition_tokens(state),
            scene_center=scene_center,
            scene_scale=scene_scale,
            world_to_scene_rotation=world_to_scene_rotation,
        )

    def encode_scene_memory(
        self,
        *,
        capture_latents: torch.Tensor,
        capture_c2w: torch.Tensor,
        capture_intrinsics: torch.Tensor,
        capture_valid_mask: torch.Tensor,
        scene_center: Optional[torch.Tensor] = None,
        scene_scale: Optional[torch.Tensor] = None,
        world_to_scene_rotation: Optional[torch.Tensor] = None,
        initial_state: Optional[RecurrentSceneMemoryState] = None,
    ) -> LongSceneMemoryEncoding:
        """Stream every capture frame into a target-independent scene state."""

        if scene_center is None or scene_scale is None:
            inferred_center, inferred_scale = compute_scene_normalization(
                capture_c2w, capture_valid_mask
            )
            scene_center = inferred_center if scene_center is None else scene_center
            scene_scale = inferred_scale if scene_scale is None else scene_scale
        if world_to_scene_rotation is None:
            world_to_scene_rotation = compute_world_to_scene_rotation(
                capture_c2w, capture_valid_mask
            )
        scene_center = scene_center.to(
            device=capture_c2w.device, dtype=capture_c2w.dtype
        )
        scene_scale = scene_scale.to(
            device=capture_c2w.device, dtype=capture_c2w.dtype
        )
        world_to_scene_rotation = world_to_scene_rotation.to(
            device=capture_c2w.device, dtype=capture_c2w.dtype
        )
        normalized_capture = normalize_camera_poses(
            capture_c2w,
            scene_center,
            scene_scale,
            world_to_scene_rotation,
        )
        state = self.scene_memory.encode_stream(
            latents=capture_latents,
            c2w=normalized_capture,
            intrinsics=capture_intrinsics,
            valid_mask=capture_valid_mask.bool(),
            chunk_size=self.long_scene_config.capture_chunk_size,
            consolidation_interval=self.long_scene_config.consolidation_interval,
            truncate_bptt_every=(
                self.long_scene_config.truncate_memory_bptt_every
            ),
            initial_state=initial_state,
        )
        return self._wrap_memory(
            state,
            scene_center=scene_center,
            scene_scale=scene_scale,
            world_to_scene_rotation=world_to_scene_rotation,
        )

    def update_scene_memory(
        self,
        memory: LongSceneMemoryEncoding,
        *,
        latents: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        valid_mask: torch.Tensor,
        source_type: int | MemorySource | torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
        consolidate: bool = False,
    ) -> tuple[LongSceneMemoryEncoding, MemoryUpdateDiagnostics]:
        """Write new capture or generated evidence into an existing state.

        ``latents`` uses observation layout ``(B,N,C,H,W)``. Callers should
        fork ``memory.state`` per output trajectory. Unverified generations
        should normally set ``consolidate=False``; verified evidence can set
        ``source_type=VERIFIED_GENERATED`` and consolidate into a shared copy.
        """

        normalized_c2w = normalize_camera_poses(
            c2w,
            memory.scene_center,
            memory.scene_scale,
            memory.world_to_scene_rotation,
        )
        output = self.scene_memory.update(
            memory.state,
            latents=latents,
            c2w=normalized_c2w,
            intrinsics=intrinsics,
            valid_mask=valid_mask.bool(),
            source_type=source_type,
            confidence=confidence,
            consolidate=consolidate,
        )
        updated = self._wrap_memory(
            output.state,
            scene_center=memory.scene_center,
            scene_scale=memory.scene_scale,
            world_to_scene_rotation=memory.world_to_scene_rotation,
        )
        return updated, output.diagnostics

    def denoise_target(
        self,
        *,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
        memory: LongSceneMemoryEncoding,
    ) -> torch.Tensor:
        batch, _, latent_frames, latent_height, latent_width = noisy_latents.shape
        patch_t, patch_h, patch_w = self.patch_size
        grid_t = latent_frames // patch_t
        grid_h = latent_height // patch_h
        grid_w = latent_width // patch_w
        if target_c2w.shape[:2] != (batch, grid_t):
            raise ValueError(
                "target_c2w must be aligned to the patchified VAE latent timeline; "
                f"expected {(batch, grid_t, 4, 4)}, got {tuple(target_c2w.shape)}"
            )
        normalized_target = normalize_camera_poses(
            target_c2w,
            memory.scene_center,
            memory.scene_scale,
            memory.world_to_scene_rotation,
        )
        camera_bias = self.camera_encoder(
            normalized_target,
            target_intrinsics,
            grid_h,
            grid_w,
        )
        return self.backbone(
            noisy_latents,
            timestep,
            prompt_embeds,
            encoder_attention_mask=prompt_attention_mask,
            video_token_bias=camera_bias,
            additional_condition_tokens=memory.condition_tokens,
            return_dict=False,
        )[0]

    def _query_head(
        self,
        *,
        head: RayQueryableGeometryHead,
        memory: LongSceneMemoryEncoding,
        query_c2w: torch.Tensor,
        query_intrinsics: torch.Tensor,
        grid_height: int,
        grid_width: int,
        max_queries: int,
        query_indices: Optional[torch.Tensor] = None,
    ) -> GeometryQueryOutput:
        normalized_query = normalize_camera_poses(
            query_c2w,
            memory.scene_center,
            memory.scene_scale,
            memory.world_to_scene_rotation,
        )
        return head(
            memory.scene_tokens,
            normalized_query,
            query_intrinsics,
            grid_height,
            grid_width,
            max_queries=max_queries,
            query_indices=query_indices,
        )

    def query_geometry(
        self,
        memory: LongSceneMemoryEncoding,
        query_c2w: torch.Tensor,
        query_intrinsics: torch.Tensor,
        grid_height: int,
        grid_width: int,
        query_indices: Optional[torch.Tensor] = None,
    ) -> GeometryQueryOutput:
        return self._query_head(
            head=self.geometry_head,
            memory=memory,
            query_c2w=query_c2w,
            query_intrinsics=query_intrinsics,
            grid_height=grid_height,
            grid_width=grid_width,
            max_queries=self.long_scene_config.max_geometry_queries,
            query_indices=query_indices,
        )

    @staticmethod
    def _prediction_to_clean(
        noisy_latents: torch.Tensor,
        predicted_velocity: torch.Tensor,
        sigmas: torch.Tensor,
        known_prefix_mask: Optional[torch.Tensor],
        clean_prefix: Optional[torch.Tensor],
    ) -> torch.Tensor:
        sigma = sigmas.view(
            -1, *([1] * (noisy_latents.ndim - 1))
        ).float()
        clean = noisy_latents.float() - sigma * predicted_velocity.float()
        if known_prefix_mask is not None and clean_prefix is not None:
            clean = torch.where(
                known_prefix_mask.bool(), clean_prefix.float(), clean
            )
        return clean

    def forward(
        self,
        *,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
        capture_latents: torch.Tensor,
        capture_c2w: torch.Tensor,
        capture_intrinsics: torch.Tensor,
        capture_valid_mask: torch.Tensor,
        scene_center: Optional[torch.Tensor] = None,
        scene_scale: Optional[torch.Tensor] = None,
        world_to_scene_rotation: Optional[torch.Tensor] = None,
        paired_noisy_latents: Optional[torch.Tensor] = None,
        paired_timestep: Optional[torch.Tensor] = None,
        paired_target_c2w: Optional[torch.Tensor] = None,
        paired_target_intrinsics: Optional[torch.Tensor] = None,
        geometry_query_c2w: Optional[torch.Tensor] = None,
        geometry_query_intrinsics: Optional[torch.Tensor] = None,
        geometry_query_grid: Optional[tuple[int, int]] = None,
        geometry_query_indices: Optional[torch.Tensor] = None,
        memory_reconstruction_query_c2w: Optional[torch.Tensor] = None,
        memory_reconstruction_query_intrinsics: Optional[torch.Tensor] = None,
        memory_reconstruction_query_grid: Optional[tuple[int, int]] = None,
        memory_reconstruction_query_indices: Optional[torch.Tensor] = None,
        target_sigmas: Optional[torch.Tensor] = None,
        target_known_prefix_mask: Optional[torch.Tensor] = None,
        target_clean_prefix: Optional[torch.Tensor] = None,
        target_write_valid_mask: Optional[torch.Tensor] = None,
        target_write_confidence: Optional[torch.Tensor] = None,
        teacher_memory_latents: Optional[torch.Tensor] = None,
        rollout_noisy_latents: Optional[torch.Tensor] = None,
        rollout_timestep: Optional[torch.Tensor] = None,
        rollout_target_c2w: Optional[torch.Tensor] = None,
        rollout_target_intrinsics: Optional[torch.Tensor] = None,
    ) -> LongSceneModelOutput:
        memory = self.encode_scene_memory(
            capture_latents=capture_latents,
            capture_c2w=capture_c2w,
            capture_intrinsics=capture_intrinsics,
            capture_valid_mask=capture_valid_mask,
            scene_center=scene_center,
            scene_scale=scene_scale,
            world_to_scene_rotation=world_to_scene_rotation,
        )
        predicted_velocity = self.denoise_target(
            noisy_latents=noisy_latents,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            target_c2w=target_c2w,
            target_intrinsics=target_intrinsics,
            memory=memory,
        )

        paired_predicted_velocity = None
        paired_arguments = (
            paired_noisy_latents,
            paired_timestep,
            paired_target_c2w,
            paired_target_intrinsics,
        )
        if any(value is not None for value in paired_arguments):
            if not all(value is not None for value in paired_arguments):
                raise ValueError(
                    "paired noisy latents, timestep, c2w, and intrinsics must "
                    "be provided together"
                )
            paired_predicted_velocity = self.denoise_target(
                noisy_latents=paired_noisy_latents,
                timestep=paired_timestep,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                target_c2w=paired_target_c2w,
                target_intrinsics=paired_target_intrinsics,
                memory=memory,
            )

        geometry = None
        geometry_arguments = (geometry_query_c2w, geometry_query_intrinsics)
        if any(value is not None for value in geometry_arguments):
            if not all(value is not None for value in geometry_arguments):
                raise ValueError("Both geometry query camera tensors are required")
            if geometry_query_grid is None:
                geometry_query_grid = (
                    noisy_latents.shape[-2] // self.patch_size[1],
                    noisy_latents.shape[-1] // self.patch_size[2],
                )
            geometry = self.query_geometry(
                memory,
                geometry_query_c2w,
                geometry_query_intrinsics,
                geometry_query_grid[0],
                geometry_query_grid[1],
                geometry_query_indices,
            )

        memory_reconstruction = None
        reconstruction_arguments = (
            memory_reconstruction_query_c2w,
            memory_reconstruction_query_intrinsics,
        )
        if any(value is not None for value in reconstruction_arguments):
            if not all(value is not None for value in reconstruction_arguments):
                raise ValueError(
                    "Both memory reconstruction camera tensors are required"
                )
            if memory_reconstruction_query_grid is None:
                memory_reconstruction_query_grid = (
                    self.long_scene_config.memory_input_grid_h,
                    self.long_scene_config.memory_input_grid_w,
                )
            memory_reconstruction = self._query_head(
                head=self.memory_readout_head,
                memory=memory,
                query_c2w=memory_reconstruction_query_c2w,
                query_intrinsics=memory_reconstruction_query_intrinsics,
                grid_height=memory_reconstruction_query_grid[0],
                grid_width=memory_reconstruction_query_grid[1],
                max_queries=(
                    self.long_scene_config.max_memory_reconstruction_queries
                ),
                query_indices=memory_reconstruction_query_indices,
            )

        updated_memory = None
        teacher_updated_memory = None
        rollout_predicted_velocity = None
        memory_update_diagnostics = None
        rollout_arguments = (
            rollout_noisy_latents,
            rollout_timestep,
            rollout_target_c2w,
            rollout_target_intrinsics,
        )
        dynamic_requested = any(value is not None for value in rollout_arguments)
        if dynamic_requested:
            if not all(value is not None for value in rollout_arguments):
                raise ValueError(
                    "rollout noisy latents, timestep, c2w, and intrinsics must "
                    "be provided together"
                )
            if target_sigmas is None:
                raise ValueError("target_sigmas are required for generated writes")
            predicted_clean = self._prediction_to_clean(
                noisy_latents,
                predicted_velocity,
                target_sigmas,
                target_known_prefix_mask,
                target_clean_prefix,
            )
            if self.long_scene_config.detach_generated_writes:
                predicted_clean = predicted_clean.detach()
            observations = predicted_clean.permute(0, 2, 1, 3, 4).to(
                capture_latents.dtype
            )
            batch, frames = observations.shape[:2]
            if target_write_valid_mask is None:
                target_write_valid_mask = torch.ones(
                    batch,
                    frames,
                    device=observations.device,
                    dtype=torch.bool,
                )
            if target_write_confidence is None:
                target_write_confidence = (
                    1.0 - target_sigmas.float()
                ).clamp(0.05, 1.0)[:, None].expand(batch, frames)
            updated_memory, memory_update_diagnostics = self.update_scene_memory(
                memory,
                latents=observations,
                c2w=target_c2w,
                intrinsics=target_intrinsics,
                valid_mask=target_write_valid_mask,
                source_type=MemorySource.GENERATED,
                confidence=target_write_confidence,
                consolidate=False,
            )
            if teacher_memory_latents is not None:
                teacher_observations = teacher_memory_latents.permute(
                    0, 2, 1, 3, 4
                ).to(capture_latents.dtype)
                teacher_updated_memory, _ = self.update_scene_memory(
                    memory,
                    latents=teacher_observations,
                    c2w=target_c2w,
                    intrinsics=target_intrinsics,
                    valid_mask=target_write_valid_mask,
                    source_type=MemorySource.GENERATED,
                    confidence=torch.ones_like(
                        target_write_confidence, dtype=observations.dtype
                    ),
                    consolidate=False,
                )
            rollout_predicted_velocity = self.denoise_target(
                noisy_latents=rollout_noisy_latents,
                timestep=rollout_timestep,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                target_c2w=rollout_target_c2w,
                target_intrinsics=rollout_target_intrinsics,
                memory=updated_memory,
            )

        return LongSceneModelOutput(
            predicted_velocity=predicted_velocity,
            memory=memory,
            paired_predicted_velocity=paired_predicted_velocity,
            geometry=geometry,
            memory_reconstruction=memory_reconstruction,
            updated_memory=updated_memory,
            teacher_updated_memory=teacher_updated_memory,
            rollout_predicted_velocity=rollout_predicted_velocity,
            memory_update_diagnostics=memory_update_diagnostics,
        )
