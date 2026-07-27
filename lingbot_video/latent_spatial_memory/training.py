from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .geometry import make_plucker_rays, scale_intrinsics
from .memory import LatentSpatialMemory, memory_consistency_mask
from .model import metric_depth_loss


@dataclass(frozen=True)
class MemoryTrainingConfig:
    latent_frames_per_chunk: int = 9
    flow_loss_weight: float = 1.0
    depth_loss_weight: float = 0.1
    teacher_memory_probability: float = 0.8
    memory_update_noise_std: float = 0.05
    memory_update_dropout_probability: float = 0.05
    timestep_shift: float = 5.0
    min_depth: float = 0.1
    max_depth: float = 20.0
    depth_edge_threshold: float = 0.08
    memory_consistency_threshold: float = 0.15
    memory_max_points: int = 750_000
    memory_voxel_size: float = 0.0
    capture_encode_batch_size: int = 2


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
    """Temporally encode capture clips and flatten their latent anchors.

    Input is ``[B,Nclip,3,F,H,W]``.  For a causal stride-4 VAE, a 9-frame
    capture clip produces three temporal latents aligned to RGB frames
    ``[0,4,8]``.  Output is ``[B,Nclip*Tlatent,C,Hl,Wl]`` in chronological
    clip/anchor order, matching the flattened pose/depth arrays from the
    dataset.
    """

    if capture_rgb.ndim == 5:
        capture_rgb = capture_rgb.unsqueeze(3)
    if capture_rgb.ndim != 6:
        raise ValueError(
            "capture_rgb must be [B,Nclip,3,F,H,W] (or legacy [B,N,3,H,W]), "
            f"got {tuple(capture_rgb.shape)}"
        )
    batch, clips, channels, frames, height, width = capture_rgb.shape
    flat = capture_rgb.reshape(batch * clips, channels, frames, height, width)
    outputs = []
    for start in range(0, flat.shape[0], micro_batch_size):
        outputs.append(
            encode_video_latents(
                vae,
                flat[start : start + micro_batch_size],
                generator=generator,
            )
        )
    encoded = torch.cat(outputs, dim=0)
    encoded = encoded.reshape(
        batch,
        clips,
        encoded.shape[1],
        encoded.shape[2],
        encoded.shape[3],
        encoded.shape[4],
    )
    return encoded.permute(0, 1, 3, 2, 4, 5).reshape(
        batch,
        clips * encoded.shape[3],
        encoded.shape[2],
        encoded.shape[4],
        encoded.shape[5],
    )


@torch.no_grad()
def encode_reference_latents(
    vae: torch.nn.Module,
    reference_rgb: torch.Tensor,
    *,
    micro_batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Encode independent clean references to ``[B,C,R,Hl,Wl]``."""

    if reference_rgb.ndim != 5:
        raise ValueError("reference_rgb must be [B,R,3,H,W]")
    batch, references, channels, height, width = reference_rgb.shape
    if references == 0:
        latent_height = height // int(getattr(vae.config, "spatial_compression_ratio", 16))
        latent_width = width // int(getattr(vae.config, "spatial_compression_ratio", 16))
        return reference_rgb.new_empty(
            batch,
            len(vae.config.latents_mean),
            0,
            latent_height,
            latent_width,
        )
    flat = reference_rgb.reshape(batch * references, channels, height, width)
    outputs = []
    for start in range(0, flat.shape[0], micro_batch_size):
        outputs.append(
            encode_video_latents(
                vae,
                flat[start : start + micro_batch_size].unsqueeze(2),
                generator=generator,
            )[:, :, 0]
        )
    encoded = torch.cat(outputs, dim=0).reshape(
        batch,
        references,
        -1,
        outputs[0].shape[-2],
        outputs[0].shape[-1],
    )
    return encoded.permute(0, 2, 1, 3, 4).contiguous()


def _flow_noise(
    clean_latents: torch.Tensor,
    *,
    shift: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _channels, frames, _height, _width = clean_latents.shape
    sigma = torch.rand(
        batch,
        frames,
        device=clean_latents.device,
        dtype=torch.float32,
    )
    sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
    # The overlapping first latent is a clean I2V anchor in MIRAGE.
    sigma[:, 0] = 0
    sigma_view = sigma[:, None, :, None, None]
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
            feature = capture_latents[batch_index, frame_index].detach()
            # During inference every completed chunk is written back and later
            # queried.  The first anchor is exact; subsequent teacher anchors
            # are stochastically corrupted/dropped so the denoiser does not
            # assume recursively generated memory is perfect.
            use_teacher = (
                frame_index == 0
                or torch.rand((), device=feature.device).item()
                < config.teacher_memory_probability
            )
            if not use_teacher and config.memory_update_noise_std > 0:
                feature = feature + torch.randn_like(feature) * float(
                    config.memory_update_noise_std
                )
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
            if (
                frame_index > 0
                and config.memory_update_dropout_probability > 0
            ):
                keep = torch.rand(
                    candidate_valid.shape,
                    device=candidate_valid.device,
                ) >= float(config.memory_update_dropout_probability)
                candidate_valid = candidate_valid & keep
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
                feature,
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
    preceding_latents = encode_video_latents(
        vae,
        batch["preceding_rgb"],
        generator=generator,
    )
    reference_latents = encode_reference_latents(
        vae,
        batch["reference_rgb"],
        micro_batch_size=config.capture_encode_batch_size,
        generator=generator,
    )
    target_latents = encode_video_latents(
        vae,
        batch["target_rgb"],
        generator=generator,
    )
    if target_latents.shape[2] != config.latent_frames_per_chunk:
        raise ValueError(
            f"expected {config.latent_frames_per_chunk} target latents, "
            f"got {target_latents.shape[2]}"
        )
    if capture_latents.shape[1] != batch["capture_c2w"].shape[1]:
        raise ValueError(
            "temporal capture encoding/geometry anchor count mismatch: "
            f"{capture_latents.shape[1]} vs {batch['capture_c2w'].shape[1]}"
        )
    if preceding_latents.shape[2] != batch["preceding_c2w"].shape[1]:
        raise ValueError(
            "preceding VAE latent/geometry anchor count mismatch: "
            f"{preceding_latents.shape[2]} vs {batch['preceding_c2w'].shape[1]}"
        )
    # Match autoregressive inference exactly: the clean overlap token is the
    # final causal latent of the preceding capture clip, not a fresh T=1 VAE
    # encoding of the same RGB frame.
    target_latents = target_latents.clone()
    target_latents[:, :, 0] = capture_latents[:, -1]
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
    preceding_intrinsics_latent = scale_intrinsics(
        batch["preceding_intrinsics"],
        image_hw,
        latent_hw,
    )
    target_rays = _plucker_video(
        batch["target_c2w"],
        target_intrinsics_latent,
        latent_hw,
    )
    preceding_rays = _plucker_video(
        batch["preceding_c2w"],
        preceding_intrinsics_latent,
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

    condition_c2w = torch.cat(
        (batch["target_c2w"], batch["preceding_c2w"]),
        dim=1,
    )
    condition_intrinsics = torch.cat(
        (target_intrinsics_latent, preceding_intrinsics_latent),
        dim=1,
    )
    memory_latents, memory_visibility, projected_depth = _read_memory_batch(
        memories,
        condition_c2w,
        condition_intrinsics,
        latent_hw,
    )

    noisy, velocity_target, sigma, sigma_view = _flow_noise(
        target_latents,
        shift=config.timestep_shift,
    )
    noisy[:, :, 0] = target_latents[:, :, 0]
    output = model(
        noisy_latents=noisy,
        target_timesteps=sigma.to(noisy.dtype) * 1000.0,
        encoder_hidden_states=embeddings,
        encoder_attention_mask=masks,
        memory_latents=memory_latents,
        memory_visibility=memory_visibility,
        target_rays=target_rays,
        preceding_latents=preceding_latents,
        preceding_rays=preceding_rays,
        reference_latents=reference_latents,
    )
    flow_loss = _masked_flow_loss(output.velocity, velocity_target)
    predicted_clean = noisy - sigma_view.to(noisy.dtype) * output.velocity
    predicted_clean = predicted_clean.clone()
    predicted_clean[:, :, 0] = target_latents[:, :, 0]
    target_frames = target_latents.shape[2]
    target_projected_depth = projected_depth[:, :, :target_frames]
    target_visibility = memory_visibility[:, :, :target_frames]
    depth_model = model.module if hasattr(model, "module") else model
    predicted_log_depth = depth_model.predict_log_depth(
        predicted_clean,
        target_rays,
        target_projected_depth,
        target_visibility,
    )
    depth_loss = metric_depth_loss(
        predicted_log_depth,
        target_depth_latent.unsqueeze(1),
        target_valid_latent.unsqueeze(1),
    )
    loss = config.flow_loss_weight * flow_loss + config.depth_loss_weight * depth_loss
    return TrainingStepOutput(
        loss=loss,
        flow_loss=flow_loss.detach(),
        depth_loss=depth_loss.detach(),
        readout_error=_visible_readout_error(
            memory_latents[:, :, :target_frames],
            target_latents,
            target_visibility,
        ).detach(),
        memory_points=sum(len(memory) for memory in memories) / len(memories),
        visible_fraction=float(target_visibility.mean().item()),
    )
