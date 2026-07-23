from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .camera import normalize_camera_poses
from .config import LongSceneConfig
from .losses import (
    add_flow_noise,
    gather_flat_queries,
    geometry_feature_cosine_loss,
    masked_flow_matching_loss,
    multi_view_reprojection_loss,
    predict_clean_latents,
    sample_flow_sigmas,
    scale_invariant_log_depth_loss,
)
from .model import LongSceneWorldModel


@dataclass
class TrainingStepOutput:
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]
    sigmas: torch.Tensor


def _default_prefix_mask(
    latents: torch.Tensor,
    prefix_frames: int,
) -> torch.Tensor:
    mask = torch.zeros(
        latents.shape[0],
        1,
        latents.shape[2],
        1,
        1,
        dtype=torch.bool,
        device=latents.device,
    )
    if prefix_frames > 0:
        mask[:, :, : min(prefix_frames, latents.shape[2])] = True
    return mask


def _prepare_noisy_target(
    clean: torch.Tensor,
    config: LongSceneConfig,
    prefix_mask: Optional[torch.Tensor],
    prefix_condition: Optional[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if prefix_mask is None:
        prefix_mask = _default_prefix_mask(clean, config.prefix_latent_frames)
    sigmas = sample_flow_sigmas(
        clean.shape[0],
        clean.device,
        clean.dtype,
        config.sigma_logit_mean,
        config.sigma_logit_std,
    )
    noisy, target_velocity, _ = add_flow_noise(
        clean, sigmas, known_prefix_mask=prefix_mask
    )
    prefix_values = clean if prefix_condition is None else prefix_condition
    if prefix_values.shape != clean.shape:
        raise ValueError(
            "prefix_condition_latents must have the same shape as target latents"
        )
    noisy = torch.where(prefix_mask.bool(), prefix_values, noisy)
    return noisy, target_velocity, sigmas, prefix_mask, prefix_values


def _resize_depth_grid(
    depth: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    batch, frames = depth.shape[:2]
    resized = F.interpolate(
        depth.reshape(batch * frames, 1, *depth.shape[-2:]).float(),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    return resized.reshape(batch, frames, height, width)


def _capture_latent_grid(
    capture_latents: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    batch, frames, channels = capture_latents.shape[:3]
    pooled = F.adaptive_avg_pool2d(
        capture_latents.reshape(
            batch * frames,
            channels,
            capture_latents.shape[-2],
            capture_latents.shape[-1],
        ).float(),
        (height, width),
    )
    return pooled.reshape(
        batch, frames, channels, height, width
    ).permute(0, 1, 3, 4, 2)


def _memory_state_consistency(
    predicted: torch.Tensor,
    teacher: torch.Tensor,
    teacher_confidence: torch.Tensor,
) -> torch.Tensor:
    teacher = teacher.detach().float()
    predicted = predicted.float()
    angular = 1.0 - F.cosine_similarity(predicted, teacher, dim=-1)
    magnitude = F.smooth_l1_loss(
        predicted, teacher, reduction="none"
    ).mean(dim=-1)
    weight = teacher_confidence.detach().float().clamp(0.0, 1.0)
    return ((angular + 0.1 * magnitude) * weight).sum() / weight.sum().clamp_min(
        1.0
    )


def _sample_valid_capture_queries(
    valid_mask: torch.Tensor,
    height: int,
    width: int,
    max_queries: int,
) -> torch.Tensor:
    valid = valid_mask.bool()[:, :, None].expand(
        -1, -1, height * width
    ).reshape(valid_mask.shape[0], -1)
    count = min(max_queries, int(valid.sum(dim=1).min().item()))
    random_scores = torch.rand(
        valid.shape, device=valid.device, dtype=torch.float32
    ).masked_fill(~valid, -1.0)
    return random_scores.topk(count, dim=1).indices


def compute_long_scene_training_loss(
    model: LongSceneWorldModel,
    batch: dict[str, torch.Tensor],
    config: LongSceneConfig,
) -> TrainingStepOutput:
    """Train capture streaming, generation, online writes, and future rollout."""

    has_paired = "paired_target_latents" in batch
    has_rollout = "rollout_target_latents" in batch
    (
        noisy,
        target_velocity,
        sigmas,
        prefix_mask,
        prefix_values,
    ) = _prepare_noisy_target(
        batch["target_latents"],
        config,
        batch.get("known_prefix_mask"),
        batch.get("prefix_condition_latents"),
    )

    paired_noisy = None
    paired_target_velocity = None
    paired_sigmas = None
    paired_prefix_mask = None
    paired_prefix_values = None
    if has_paired:
        (
            paired_noisy,
            paired_target_velocity,
            paired_sigmas,
            paired_prefix_mask,
            paired_prefix_values,
        ) = _prepare_noisy_target(
            batch["paired_target_latents"],
            config,
            batch.get("paired_known_prefix_mask"),
            batch.get("paired_prefix_condition_latents"),
        )

    rollout_noisy = None
    rollout_target_velocity = None
    rollout_sigmas = None
    rollout_prefix_mask = None
    rollout_prefix_values = None
    if has_rollout:
        (
            rollout_noisy,
            rollout_target_velocity,
            rollout_sigmas,
            rollout_prefix_mask,
            rollout_prefix_values,
        ) = _prepare_noisy_target(
            batch["rollout_target_latents"],
            config,
            batch.get("rollout_known_prefix_mask"),
            batch.get("rollout_prefix_condition_latents"),
        )

    write_confidence = (1.0 - sigmas.float()).square()[:, None].expand(
        -1, batch["target_latents"].shape[2]
    )
    known_frames = prefix_mask[:, 0, :, 0, 0].bool()
    write_confidence = torch.where(
        known_frames, torch.ones_like(write_confidence), write_confidence
    )
    memory_query_indices = _sample_valid_capture_queries(
        batch["capture_valid_mask"],
        config.memory_input_grid_h,
        config.memory_input_grid_w,
        config.max_memory_reconstruction_queries,
    )

    model_output = model(
        noisy_latents=noisy,
        timestep=sigmas.float() * 1000.0,
        prompt_embeds=batch["prompt_embeds"],
        prompt_attention_mask=batch["prompt_attention_mask"],
        target_c2w=batch["target_c2w"],
        target_intrinsics=batch["target_intrinsics"],
        capture_latents=batch["capture_latents"],
        capture_c2w=batch["capture_c2w"],
        capture_intrinsics=batch["capture_intrinsics"],
        capture_valid_mask=batch["capture_valid_mask"].bool(),
        scene_center=batch.get("scene_center"),
        scene_scale=batch.get("scene_scale"),
        world_to_scene_rotation=batch.get("world_to_scene_rotation"),
        paired_noisy_latents=paired_noisy,
        paired_timestep=(
            None if paired_sigmas is None else paired_sigmas.float() * 1000.0
        ),
        paired_target_c2w=batch.get("paired_target_c2w"),
        paired_target_intrinsics=batch.get("paired_target_intrinsics"),
        geometry_query_c2w=batch["target_c2w"],
        geometry_query_intrinsics=batch["target_intrinsics"],
        memory_reconstruction_query_c2w=batch["capture_c2w"],
        memory_reconstruction_query_intrinsics=batch["capture_intrinsics"],
        memory_reconstruction_query_indices=memory_query_indices,
        target_sigmas=sigmas if has_rollout else None,
        target_known_prefix_mask=prefix_mask if has_rollout else None,
        target_clean_prefix=prefix_values if has_rollout else None,
        target_write_valid_mask=batch.get("target_write_valid_mask"),
        target_write_confidence=write_confidence if has_rollout else None,
        teacher_memory_latents=(
            batch["target_latents"] if has_rollout else None
        ),
        rollout_noisy_latents=rollout_noisy,
        rollout_timestep=(
            None
            if rollout_sigmas is None
            else rollout_sigmas.float() * 1000.0
        ),
        rollout_target_c2w=batch.get("rollout_target_c2w"),
        rollout_target_intrinsics=batch.get("rollout_target_intrinsics"),
    )
    memory = model_output.memory
    geometry = model_output.geometry
    if geometry is None:
        raise RuntimeError("The training forward did not return geometry queries")

    flow_loss = masked_flow_matching_loss(
        model_output.predicted_velocity, target_velocity, prefix_mask
    )
    predicted_clean = predict_clean_latents(
        noisy,
        model_output.predicted_velocity,
        sigmas,
        known_prefix_mask=prefix_mask,
        clean_prefix=prefix_values,
    )
    losses: dict[str, torch.Tensor] = {"flow": flow_loss}

    if has_paired:
        if (
            model_output.paired_predicted_velocity is None
            or paired_target_velocity is None
            or paired_sigmas is None
            or paired_prefix_mask is None
            or paired_prefix_values is None
            or paired_noisy is None
        ):
            raise RuntimeError("The paired training forward is incomplete")
        paired_flow = masked_flow_matching_loss(
            model_output.paired_predicted_velocity,
            paired_target_velocity,
            paired_prefix_mask,
        )
        paired_clean = predict_clean_latents(
            paired_noisy,
            model_output.paired_predicted_velocity,
            paired_sigmas,
            known_prefix_mask=paired_prefix_mask,
            clean_prefix=paired_prefix_values,
        )
        losses["flow"] = 0.5 * (losses["flow"] + paired_flow)
    else:
        paired_clean = None

    if has_rollout:
        if (
            model_output.rollout_predicted_velocity is None
            or model_output.updated_memory is None
            or model_output.teacher_updated_memory is None
            or rollout_target_velocity is None
            or rollout_prefix_mask is None
        ):
            raise RuntimeError("The dynamic rollout training forward is incomplete")
        losses["rollout_flow"] = masked_flow_matching_loss(
            model_output.rollout_predicted_velocity,
            rollout_target_velocity,
            rollout_prefix_mask,
        )
        losses["memory_state_consistency"] = _memory_state_consistency(
            model_output.updated_memory.state.fast_tokens,
            model_output.teacher_updated_memory.state.fast_tokens,
            model_output.teacher_updated_memory.state.fast_confidence,
        )

    reconstruction = model_output.memory_reconstruction
    if reconstruction is None or reconstruction.features is None:
        raise RuntimeError("The training forward did not return memory readout")
    _, reconstruction_height, reconstruction_width = reconstruction.query_grid
    capture_grid = _capture_latent_grid(
        batch["capture_latents"],
        reconstruction_height,
        reconstruction_width,
    )
    capture_targets = gather_flat_queries(
        capture_grid, reconstruction.query_indices
    )
    valid_grid = batch["capture_valid_mask"].bool()[:, :, None, None].expand(
        -1, -1, reconstruction_height, reconstruction_width
    )
    capture_query_valid = gather_flat_queries(
        valid_grid.float(), reconstruction.query_indices
    )
    losses["memory_reconstruction"] = geometry_feature_cosine_loss(
        reconstruction.features,
        capture_targets,
        capture_query_valid,
        scale_weight=1.0,
    )

    _, query_height, query_width = geometry.query_grid
    normalized_depth = batch["target_depth"].float() / memory.scene_scale[
        :, None, None, None
    ].float()
    depth_grid = _resize_depth_grid(
        normalized_depth, query_height, query_width
    )
    sampled_depth = gather_flat_queries(depth_grid, geometry.query_indices)
    sampled_confidence = None
    if "target_depth_confidence" in batch:
        confidence_grid = _resize_depth_grid(
            batch["target_depth_confidence"], query_height, query_width
        )
        sampled_confidence = gather_flat_queries(
            confidence_grid, geometry.query_indices
        )
    losses["geometry_depth"] = scale_invariant_log_depth_loss(
        geometry.log_depth,
        sampled_depth,
        sampled_confidence,
        scale_weight=config.geometry_depth_scale_weight,
    )

    if geometry.features is not None:
        if "geometry_teacher_features" not in batch:
            raise ValueError(
                "geometry_teacher_features are required when "
                "geometry_feature_dim is non-zero"
            )
        teacher_features = batch["geometry_teacher_features"]
        if teacher_features.shape[2:4] != (query_height, query_width):
            raise ValueError(
                "geometry_teacher_features must already match the DiT patch grid"
            )
        sampled_features = gather_flat_queries(
            teacher_features, geometry.query_indices
        )
        losses["geometry_feature"] = geometry_feature_cosine_loss(
            geometry.features,
            sampled_features,
            sampled_confidence,
            scale_weight=config.geometry_feature_scale_weight,
        )

    geometry_gate = (sigmas <= config.geometry_loss_max_sigma).float().mean()
    if "consistency_pairs" in batch:
        normalized_target_c2w = normalize_camera_poses(
            batch["target_c2w"],
            memory.scene_center,
            memory.scene_scale,
            memory.world_to_scene_rotation,
        )
        losses["reprojection"] = geometry_gate * multi_view_reprojection_loss(
            predicted_clean,
            normalized_depth,
            normalized_target_c2w,
            batch["target_intrinsics"],
            batch["consistency_pairs"].long(),
            batch.get("consistency_pair_valid_mask"),
            batch.get("target_depth_confidence"),
            depth_is_ray_distance=config.depth_is_ray_distance,
            occlusion_relative_threshold=config.occlusion_relative_threshold,
            occlusion_absolute_threshold=config.occlusion_absolute_threshold,
        )

    if (
        has_paired
        and paired_clean is not None
        and "cross_trajectory_pairs" in batch
    ):
        paired_depth = batch["paired_target_depth"].float() / memory.scene_scale[
            :, None, None, None
        ].float()
        combined_clean = torch.cat((predicted_clean, paired_clean), dim=2)
        combined_depth = torch.cat((normalized_depth, paired_depth), dim=1)
        combined_c2w = normalize_camera_poses(
            torch.cat(
                (batch["target_c2w"], batch["paired_target_c2w"]), dim=1
            ),
            memory.scene_center,
            memory.scene_scale,
            memory.world_to_scene_rotation,
        )
        combined_intrinsics = torch.cat(
            (
                batch["target_intrinsics"],
                batch["paired_target_intrinsics"],
            ),
            dim=1,
        )
        combined_confidence = None
        if (
            "target_depth_confidence" in batch
            and "paired_target_depth_confidence" in batch
        ):
            combined_confidence = torch.cat(
                (
                    batch["target_depth_confidence"],
                    batch["paired_target_depth_confidence"],
                ),
                dim=1,
            )
        paired_gate = geometry_gate
        if paired_sigmas is not None:
            paired_gate = 0.5 * (
                geometry_gate
                + (paired_sigmas <= config.geometry_loss_max_sigma).float().mean()
            )
        losses["paired_trajectory"] = (
            paired_gate
            * multi_view_reprojection_loss(
                combined_clean,
                combined_depth,
                combined_c2w,
                combined_intrinsics,
                batch["cross_trajectory_pairs"].long(),
                batch.get("cross_trajectory_pair_valid_mask"),
                combined_confidence,
                depth_is_ray_distance=config.depth_is_ray_distance,
                occlusion_relative_threshold=config.occlusion_relative_threshold,
                occlusion_absolute_threshold=config.occlusion_absolute_threshold,
            )
        )

    weighted = config.flow_loss_weight * losses["flow"]
    weight_map = {
        "rollout_flow": config.rollout_flow_loss_weight,
        "memory_state_consistency": (
            config.memory_state_consistency_loss_weight
        ),
        "memory_reconstruction": config.memory_reconstruction_loss_weight,
        "geometry_depth": config.geometry_depth_loss_weight,
        "geometry_feature": config.geometry_feature_loss_weight,
        "reprojection": config.reprojection_loss_weight,
        "paired_trajectory": config.paired_trajectory_loss_weight,
    }
    for name, weight in weight_map.items():
        if name in losses:
            weighted = weighted + weight * losses[name]
    losses["total"] = weighted
    return TrainingStepOutput(loss=weighted, losses=losses, sigmas=sigmas)
