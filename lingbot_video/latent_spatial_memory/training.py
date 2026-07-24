from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .geometry import make_plucker_rays, scale_intrinsics
from .memory import LatentSpatialMemory, memory_consistency_mask
from .model import PRECEDING_SEGMENT, TARGET_SEGMENT, metric_depth_loss


@dataclass(frozen=True)
class MemoryTrainingConfig:
    rollout_chunks: int = 2
    latent_frames_per_chunk: int = 9
    flow_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    teacher_memory_probability: float = 0.5
    min_depth: float = 0.1
    max_depth: float = 20.0
    depth_edge_threshold: float = 0.08
    memory_consistency_threshold: float = 0.15
    memory_max_points: int = 750_000
    memory_voxel_size: float = 0.0
    capture_encode_batch_size: int = 8


@dataclass
class TrainingStepOutput:
    loss: torch.Tensor
    flow_loss: torch.Tensor
    depth_loss: torch.Tensor
    readout_error: torch.Tensor
    memory_points: float
    visible_fraction: float


def _vae_latent_to_dit(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        vae.config.latents_mean,
        device=latents.device,
        dtype=torch.float32,
    ).view(1, -1, 1, 1, 1)
    std_inv = (
        1.0
        / torch.tensor(
            vae.config.latents_std,
            device=latents.device,
            dtype=torch.float32,
        )
    ).view(1, -1, 1, 1, 1)
    return ((latents.float() - mean) * std_inv).to(latents.dtype)


@torch.no_grad()
def encode_video_latents(
    vae: torch.nn.Module,
    video: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode [B,3,F,H,W] RGB in [0,1] into LingBot DiT latents."""

    normalized = video.float().mul(2.0).sub(1.0)
    encoded = vae.encode(normalized)
    if hasattr(encoded, "latent_dist"):
        latents = encoded.latent_dist.sample(generator)
    elif isinstance(encoded, tuple):
        latents = encoded[0]
    else:
        latents = encoded
    return _vae_latent_to_dit(vae, latents)


@torch.no_grad()
def encode_capture_latents(
    vae: torch.nn.Module,
    capture_rgb: torch.Tensor,
    *,
    micro_batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode independent cache frames; output is [B,N,C,Hl,Wl]."""

    batch, frames, channels, height, width = capture_rgb.shape
    flat = capture_rgb.reshape(batch * frames, channels, height, width)
    outputs = []
    for start in range(0, flat.shape[0], micro_batch_size):
        clip = flat[start : start + micro_batch_size].unsqueeze(2)
        outputs.append(encode_video_latents(vae, clip, generator=generator)[:, :, 0])
    encoded = torch.cat(outputs, dim=0)
    return encoded.reshape(batch, frames, encoded.shape[1], encoded.shape[2], encoded.shape[3])


def _flow_noise(
    clean_latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = clean_latents.shape[0]
    sigma = torch.sigmoid(
        torch.randn(batch, device=clean_latents.device, dtype=torch.float32)
    )
    sigma_view = sigma.view(batch, 1, 1, 1, 1)
    noise = torch.randn_like(clean_latents)
    noisy = (1.0 - sigma_view) * clean_latents + sigma_view * noise
    velocity = noise - clean_latents
    return noisy, velocity, sigma, sigma_view


def _downsample_depth_and_valid(
    depth: torch.Tensor,
    valid: torch.Tensor,
    latent_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames, height, width = depth.shape
    depth_latent = F.interpolate(
        depth.reshape(batch * frames, 1, height, width).float(),
        size=latent_hw,
        mode="bilinear",
        align_corners=False,
    ).reshape(batch, frames, *latent_hw)
    valid_latent = F.interpolate(
        valid.reshape(batch * frames, 1, height, width).float(),
        size=latent_hw,
        mode="nearest",
    ).reshape(batch, frames, *latent_hw)
    valid_latent = valid_latent > 0.5
    return depth_latent, valid_latent


@torch.no_grad()
def build_capture_memories(
    capture_latents: torch.Tensor,
    batch: dict[str, torch.Tensor],
    *,
    image_hw: tuple[int, int],
    config: MemoryTrainingConfig,
) -> list[LatentSpatialMemory]:
    memories = []
    for batch_index in range(capture_latents.shape[0]):
        memory = LatentSpatialMemory(
            capture_latents.shape[2],
            device=capture_latents.device,
            feature_dtype=capture_latents.dtype,
            max_points=config.memory_max_points,
            voxel_size=config.memory_voxel_size,
        )
        for frame_index in range(capture_latents.shape[1]):
            latent_hw = capture_latents.shape[-2:]
            candidate_depth, candidate_valid = _downsample_depth_and_valid(
                batch["capture_depth"][
                    batch_index : batch_index + 1,
                    frame_index : frame_index + 1,
                ],
                batch["capture_valid"][
                    batch_index : batch_index + 1,
                    frame_index : frame_index + 1,
                ],
                latent_hw,
            )
            candidate_depth = candidate_depth[0, 0]
            candidate_valid = candidate_valid[0, 0]
            if len(memory):
                k_latent = scale_intrinsics(
                    batch["capture_intrinsics"][batch_index, frame_index],
                    image_hw,
                    latent_hw,
                )
                existing = memory.read(
                    batch["capture_c2w"][batch_index, frame_index],
                    k_latent,
                    latent_hw,
                )
                candidate_valid = memory_consistency_mask(
                    candidate_depth,
                    candidate_valid,
                    existing.depth[0, 0],
                    existing.visibility[0, 0],
                    relative_threshold=config.memory_consistency_threshold,
                )
            memory.write(
                capture_latents[batch_index, frame_index].detach(),
                batch["capture_depth"][batch_index, frame_index],
                batch["capture_intrinsics"][batch_index, frame_index],
                batch["capture_c2w"][batch_index, frame_index],
                image_hw=image_hw,
                valid_mask=candidate_valid,
                frame_id=int(batch["capture_indices"][batch_index, frame_index].item()),
                min_depth=config.min_depth,
                max_depth=config.max_depth,
                relative_edge_threshold=config.depth_edge_threshold,
                compact=False,
            )
        memory.compact()
        memories.append(memory)
    return memories


def _read_memory_batch(
    memories: list[LatentSpatialMemory],
    target_c2w: torch.Tensor,
    target_intrinsics_latent: torch.Tensor,
    latent_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    readouts = [
        memory.read(
            target_c2w[index],
            target_intrinsics_latent[index],
            latent_hw,
        )
        for index, memory in enumerate(memories)
    ]
    return (
        torch.stack([readout.features for readout in readouts]),
        torch.stack([readout.visibility for readout in readouts]),
        torch.stack([readout.depth for readout in readouts]),
    )


def _plucker_video(
    c2w: torch.Tensor,
    intrinsics_latent: torch.Tensor,
    latent_hw: tuple[int, int],
) -> torch.Tensor:
    height, width = latent_hw
    # make_plucker_rays returns [B,T,6,H,W].
    return make_plucker_rays(c2w, intrinsics_latent, height, width).transpose(1, 2)


def _masked_flow_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    # The first latent is the clean overlap with the preceding chunk.
    if predicted.shape[2] <= 1:
        return predicted.sum() * 0
    return F.mse_loss(predicted[:, :, 1:].float(), target[:, :, 1:].float())


def _visible_readout_error(
    readout: torch.Tensor,
    clean: torch.Tensor,
    visibility: torch.Tensor,
) -> torch.Tensor:
    valid = visibility.expand_as(readout) > 0.5
    if not valid.any():
        return readout.sum() * 0
    return F.smooth_l1_loss(readout.float()[valid], clean.float()[valid], beta=0.1)


def latent_memory_training_step(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    config: MemoryTrainingConfig,
    *,
    text_dropout_probability: float = 0.2,
    generator: Optional[torch.Generator] = None,
) -> TrainingStepOutput:
    device = batch["target_rgb"].device
    image_hw = tuple(int(value) for value in batch["image_hw"][0].tolist())
    capture_latents = encode_capture_latents(
        vae,
        batch["capture_rgb"],
        micro_batch_size=config.capture_encode_batch_size,
        generator=generator,
    )
    target_latents = encode_video_latents(
        vae,
        batch["target_rgb"],
        generator=generator,
    )
    expected_frames = 1 + config.rollout_chunks * (config.latent_frames_per_chunk - 1)
    if target_latents.shape[2] != expected_frames:
        raise ValueError(
            f"expected {expected_frames} target latents, got {target_latents.shape[2]}"
        )
    latent_hw = (target_latents.shape[-2], target_latents.shape[-1])
    memories = build_capture_memories(
        capture_latents,
        batch,
        image_hw=image_hw,
        config=config,
    )

    target_intrinsics_latent = scale_intrinsics(
        batch["target_intrinsics"],
        image_hw,
        latent_hw,
    )
    capture_intrinsics_latent = scale_intrinsics(
        batch["capture_intrinsics"],
        image_hw,
        latent_hw,
    )
    target_rays_all = _plucker_video(
        batch["target_c2w"],
        target_intrinsics_latent,
        latent_hw,
    )
    reference_latents = capture_latents[:, 0].unsqueeze(2)
    reference_rays = _plucker_video(
        batch["capture_c2w"][:, :1],
        capture_intrinsics_latent[:, :1],
        latent_hw,
    )
    target_depth_latent, target_valid_latent = _downsample_depth_and_valid(
        batch["target_depth"],
        batch["target_valid"],
        latent_hw,
    )

    embeddings = prompt_embeds.expand(target_latents.shape[0], -1, -1)
    masks = prompt_mask.expand(target_latents.shape[0], -1)
    if text_dropout_probability > 0:
        dropped = (
            torch.rand(target_latents.shape[0], device=device)
            < text_dropout_probability
        )
        if dropped.any():
            embeddings = embeddings.clone()
            embeddings[dropped] = 0

    flow_losses = []
    depth_losses = []
    readout_errors = []
    visible_fractions = []
    preceding_latent: Optional[torch.Tensor] = None
    stride = config.latent_frames_per_chunk - 1
    for chunk_index in range(config.rollout_chunks):
        start = chunk_index * stride
        end = start + config.latent_frames_per_chunk
        clean = target_latents[:, :, start:end]
        chunk_c2w = batch["target_c2w"][:, start:end]
        chunk_intrinsics = target_intrinsics_latent[:, start:end]
        chunk_rays = target_rays_all[:, :, start:end]
        memory_latents, memory_visibility, projected_depth = _read_memory_batch(
            memories,
            chunk_c2w,
            chunk_intrinsics,
            latent_hw,
        )

        noisy, velocity_target, sigma, sigma_view = _flow_noise(clean)
        if preceding_latent is None:
            preceding_latent = clean[:, :, 0].detach()
        noisy[:, :, 0] = preceding_latent
        segment_ids = torch.full(
            (clean.shape[0], clean.shape[2]),
            TARGET_SEGMENT,
            device=device,
            dtype=torch.long,
        )
        segment_ids[:, 0] = PRECEDING_SEGMENT
        output = model(
            noisy_latents=noisy,
            timestep=sigma.to(noisy.dtype) * 1000.0,
            encoder_hidden_states=embeddings,
            encoder_attention_mask=masks,
            memory_latents=memory_latents,
            memory_visibility=memory_visibility,
            target_rays=chunk_rays,
            target_segment_ids=segment_ids,
            reference_latents=reference_latents,
            reference_rays=reference_rays,
        )
        flow_losses.append(_masked_flow_loss(output.velocity, velocity_target))
        predicted_clean = noisy - sigma_view.to(noisy.dtype) * output.velocity
        predicted_clean = predicted_clean.clone()
        predicted_clean[:, :, 0] = preceding_latent
        depth_model = model.module if hasattr(model, "module") else model
        predicted_log_depth = depth_model.predict_log_depth(
            predicted_clean,
            chunk_rays,
            projected_depth,
            memory_visibility,
        )
        chunk_depth = target_depth_latent[:, start:end].unsqueeze(1)
        chunk_valid = target_valid_latent[:, start:end].unsqueeze(1)
        depth_losses.append(
            metric_depth_loss(predicted_log_depth, chunk_depth, chunk_valid)
        )
        readout_errors.append(
            _visible_readout_error(memory_latents, clean, memory_visibility).detach()
        )
        visible_fractions.append(memory_visibility.mean().detach())

        with torch.no_grad():
            predicted_depth = predicted_log_depth.exp().squeeze(1)
            for batch_index, memory in enumerate(memories):
                use_teacher = (
                    torch.rand((), device=device).item()
                    < config.teacher_memory_probability
                )
                update_latents = (
                    clean[batch_index, :, 1:].detach()
                    if use_teacher
                    else predicted_clean[batch_index, :, 1:].detach()
                )
                update_depth = (
                    target_depth_latent[batch_index, start + 1 : end]
                    if use_teacher
                    else predicted_depth[batch_index, 1:]
                )
                update_valid = (
                    target_valid_latent[batch_index, start + 1 : end]
                    if use_teacher
                    else torch.isfinite(update_depth)
                    & (update_depth >= config.min_depth)
                    & (update_depth <= config.max_depth)
                )
                update_valid = memory_consistency_mask(
                    update_depth,
                    update_valid,
                    projected_depth[batch_index, 0, 1:],
                    memory_visibility[batch_index, 0, 1:],
                    relative_threshold=config.memory_consistency_threshold,
                )
                memory.write_video(
                    update_latents,
                    update_depth,
                    chunk_intrinsics[batch_index, 1:],
                    chunk_c2w[batch_index, 1:],
                    image_hw=latent_hw,
                    valid_masks=update_valid,
                    frame_ids=batch["target_latent_indices"][
                        batch_index, start + 1 : end
                    ],
                    min_depth=config.min_depth,
                    max_depth=config.max_depth,
                    relative_edge_threshold=config.depth_edge_threshold,
                )
        preceding_latent = predicted_clean[:, :, -1].detach()

    flow_loss = torch.stack(flow_losses).mean()
    depth_loss = torch.stack(depth_losses).mean()
    loss = config.flow_loss_weight * flow_loss + config.depth_loss_weight * depth_loss
    return TrainingStepOutput(
        loss=loss,
        flow_loss=flow_loss.detach(),
        depth_loss=depth_loss.detach(),
        readout_error=torch.stack(readout_errors).mean(),
        memory_points=sum(len(memory) for memory in memories) / len(memories),
        visible_fraction=float(torch.stack(visible_fractions).mean().item()),
    )
