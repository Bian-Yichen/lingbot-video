from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .data import RoomTourSample, VipeRoomTourItem
from .model import GIMWorldLingBotModel
from .pruning import MIGreedyPruner
from .teacher import VGGTGeometryTeacher


@dataclass(frozen=True)
class GIMTrainingConfig:
    pruning_budget: int = 200
    geometry_loss_weight: float = 0.05
    vae_temporal_stride: int = 4
    vae_encode_chunk_rgb_frames: int = 81
    timestep_shift: float = 1.0
    predicted_update_probability: float = 0.0


@dataclass
class PreparedQueryBlock:
    rgb_indices: tuple[int, ...]
    latent_rgb_indices: tuple[int, ...]
    latents: torch.Tensor
    c2w: torch.Tensor
    intrinsics: torch.Tensor
    geometry_query_index: int
    geometry_query_rgb: torch.Tensor
    geometry_query_c2w: torch.Tensor
    geometry_query_intrinsics: torch.Tensor


@dataclass
class PreparedTrajectoryBatch:
    item: VipeRoomTourItem
    sample: RoomTourSample
    capture_latents: torch.Tensor
    capture_c2w: torch.Tensor
    capture_intrinsics: torch.Tensor
    capture_times: torch.Tensor
    query_blocks: list[PreparedQueryBlock]


def _vae_latent_to_dit(
    vae: torch.nn.Module,
    latents: torch.Tensor,
) -> torch.Tensor:
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


def _require_streamable_wan_vae(
    vae: torch.nn.Module,
    temporal_stride: int,
) -> None:
    # AutoencoderKLWan creates its encoder feature-state attributes lazily in
    # clear_cache().  They therefore must not be required immediately after
    # from_pretrained(), before clear_cache() has run for the first time.
    missing = [
        name
        for name in (
            "clear_cache",
            "encoder",
            "quant_conv",
        )
        if not hasattr(vae, name)
    ]
    if missing:
        raise TypeError(
            "online trajectory encoding requires a compatible diffusers "
            "AutoencoderKLWan causal encoder; "
            f"missing attributes: {missing}"
        )
    configured_stride = int(
        getattr(vae.config, "scale_factor_temporal", temporal_stride)
    )
    if configured_stride != temporal_stride:
        raise ValueError(
            "configured VAE temporal compression does not match the dataset "
            f"timeline: VAE={configured_stride}, requested={temporal_stride}"
        )
    if getattr(vae.config, "patch_size", None) is not None:
        raise NotImplementedError(
            "online trajectory encoding does not support a patchified Wan VAE"
        )


def _reset_wan_encoder_state(vae: torch.nn.Module) -> None:
    """Initialize a fresh causal encoder state for one trajectory."""

    vae.clear_cache()
    missing = [
        name
        for name in ("_enc_feat_map", "_enc_conv_idx")
        if not hasattr(vae, name)
    ]
    if missing:
        raise TypeError(
            "AutoencoderKLWan.clear_cache() did not initialize the causal "
            f"encoder state required for streaming: {missing}"
        )
    if not isinstance(vae._enc_feat_map, list) or not isinstance(
        vae._enc_conv_idx, list
    ):
        raise TypeError(
            "AutoencoderKLWan causal encoder state has an unsupported type: "
            f"_enc_feat_map={type(vae._enc_feat_map).__name__}, "
            f"_enc_conv_idx={type(vae._enc_conv_idx).__name__}"
        )


@torch.no_grad()
def encode_wan_scene_streaming(
    vae: torch.nn.Module,
    item: VipeRoomTourItem,
    indices: list[int] | tuple[int, ...],
    target_hw: tuple[int, int],
    *,
    temporal_stride: int,
    read_chunk_rgb_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode one continuous RGB trajectory with one fresh Wan causal state."""

    _require_streamable_wan_vae(vae, temporal_stride)
    indices = [int(index) for index in indices]
    if not indices:
        raise ValueError("cannot encode an empty trajectory")
    if (len(indices) - 1) % temporal_stride:
        raise ValueError(
            "trajectory length must equal 1 + k * temporal_stride"
        )
    if (
        read_chunk_rgb_frames <= temporal_stride
        or (read_chunk_rgb_frames - 1) % temporal_stride
    ):
        raise ValueError(
            "read_chunk_rgb_frames must equal 1 + k * temporal_stride "
            "with k >= 1"
        )
    autocast = (
        torch.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        else contextlib.nullcontext()
    )
    latent_chunks: list[torch.Tensor] = []
    _reset_wan_encoder_state(vae)
    position = 0
    first_read = True
    try:
        while position < len(indices):
            read_count = (
                read_chunk_rgb_frames
                if first_read
                else read_chunk_rgb_frames - 1
            )
            read_count = min(read_count, len(indices) - position)
            read_indices = indices[position : position + read_count]
            video = item.read_video(read_indices, target_hw)
            local_position = 0
            while local_position < video.shape[1]:
                group_size = (
                    1
                    if position == 0 and local_position == 0
                    else temporal_stride
                )
                group = video[
                    :,
                    local_position : local_position + group_size,
                ]
                if group.shape[1] != group_size:
                    raise RuntimeError(
                        "Wan causal group ended with an incomplete frame group"
                    )
                group = (
                    group.unsqueeze(0)
                    .to(device=device, dtype=torch.float32)
                    .mul(2.0)
                    .sub(1.0)
                )
                vae._enc_conv_idx = [0]
                with autocast:
                    features = vae.encoder(
                        group,
                        feat_cache=vae._enc_feat_map,
                        feat_idx=vae._enc_conv_idx,
                    )
                    moments = vae.quant_conv(features)
                latents = moments.chunk(2, dim=1)[0]
                latent_chunks.append(
                    _vae_latent_to_dit(vae, latents).cpu()
                )
                local_position += group_size
                del group, features, moments, latents
            position += read_count
            first_read = False
            del video
    finally:
        vae.clear_cache()
    output = torch.cat(latent_chunks, dim=2)
    expected = 1 + (len(indices) - 1) // temporal_stride
    if output.shape[2] != expected:
        raise RuntimeError(
            f"Wan returned {output.shape[2]} latents, expected {expected}"
        )
    return output


@torch.no_grad()
def prepare_gim_trajectory_online(
    sample: RoomTourSample,
    *,
    vae: torch.nn.Module,
    vae_temporal_stride: int,
    vae_encode_chunk_rgb_frames: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> PreparedTrajectoryBatch:
    """Read RGB and run every VAE encoding online for one scene iteration."""

    item = VipeRoomTourItem(Path(sample.local_root))
    origin_index = sample.capture_start
    capture_latents = encode_wan_scene_streaming(
        vae,
        item,
        sample.capture_rgb_indices,
        sample.image_hw,
        temporal_stride=vae_temporal_stride,
        read_chunk_rgb_frames=vae_encode_chunk_rgb_frames,
        device=device,
        dtype=compute_dtype,
    )
    capture_latent_indices = sample.capture_rgb_indices[
        ::vae_temporal_stride
    ]
    capture_c2w, capture_k = item.cameras(
        capture_latent_indices,
        sample.image_hw,
        origin_index=origin_index,
    )
    capture_times = torch.tensor(
        [
            int(index) - origin_index
            for index in capture_latent_indices
        ],
        dtype=torch.long,
    )

    query_blocks: list[PreparedQueryBlock] = []
    for rgb_indices, geometry_query_index in zip(
        sample.query_rgb_blocks,
        sample.geometry_query_indices,
        strict=True,
    ):
        latents = encode_wan_scene_streaming(
            vae,
            item,
            rgb_indices,
            sample.image_hw,
            temporal_stride=vae_temporal_stride,
            read_chunk_rgb_frames=vae_encode_chunk_rgb_frames,
            device=device,
            dtype=compute_dtype,
        )
        latent_rgb_indices = rgb_indices[::vae_temporal_stride]
        c2w, intrinsics = item.cameras(
            latent_rgb_indices,
            sample.image_hw,
            origin_index=origin_index,
        )
        query_c2w, query_intrinsics = item.cameras(
            [geometry_query_index],
            sample.image_hw,
            origin_index=origin_index,
        )
        query_blocks.append(
            PreparedQueryBlock(
                rgb_indices=rgb_indices,
                latent_rgb_indices=latent_rgb_indices,
                latents=latents,
                c2w=c2w.unsqueeze(0),
                intrinsics=intrinsics.unsqueeze(0),
                geometry_query_index=geometry_query_index,
                geometry_query_rgb=item.read_rgb(
                    geometry_query_index,
                    sample.image_hw,
                ).unsqueeze(0),
                geometry_query_c2w=query_c2w.unsqueeze(0),
                geometry_query_intrinsics=query_intrinsics.unsqueeze(0),
            )
        )
    return PreparedTrajectoryBatch(
        item=item,
        sample=sample,
        capture_latents=capture_latents,
        capture_c2w=capture_c2w.unsqueeze(0),
        capture_intrinsics=capture_k.unsqueeze(0),
        capture_times=capture_times,
        query_blocks=query_blocks,
    )


def shifted_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _select_history(
    latents: torch.Tensor,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    times: torch.Tensor,
    *,
    pruner: MIGreedyPruner,
    budget: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = pruner.select(c2w[0].cpu(), times.cpu(), budget)
    selected = selected[
        torch.argsort(times.cpu()[selected], stable=True)
    ]
    return (
        latents[:, :, selected].to(device=device, dtype=dtype),
        c2w[:, selected].to(device),
        intrinsics[:, selected].to(device),
        selected,
    )


def gim_trajectory_training_step(
    model: GIMWorldLingBotModel,
    batch: PreparedTrajectoryBatch,
    *,
    teacher: VGGTGeometryTeacher,
    pruner: MIGreedyPruner,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    config: GIMTrainingConfig,
    device: torch.device,
    compute_dtype: torch.dtype,
    backward: Callable[[torch.Tensor], None] | None = None,
) -> dict[str, torch.Tensor]:
    """Train query blocks, updating memory only between supervised blocks."""

    history_latents = batch.capture_latents
    history_c2w = batch.capture_c2w
    history_intrinsics = batch.capture_intrinsics
    history_times = batch.capture_times
    losses: list[torch.Tensor] = []
    flow_losses: list[torch.Tensor] = []
    geometry_losses: list[torch.Tensor] = []
    sigmas: list[torch.Tensor] = []
    memory_norms: list[torch.Tensor] = []
    candidate_counts: list[torch.Tensor] = []
    retained_counts: list[torch.Tensor] = []
    predicted_updates: list[torch.Tensor] = []

    for block_index, block in enumerate(batch.query_blocks):
        history_candidates_before = history_latents.shape[2]
        selected_latents, selected_c2w, selected_k, selected = _select_history(
            history_latents,
            history_c2w,
            history_intrinsics,
            history_times,
            pruner=pruner,
            budget=config.pruning_budget,
            device=device,
            dtype=compute_dtype,
        )
        target = block.latents.to(device=device, dtype=compute_dtype)
        target_c2w = block.c2w.to(device)
        target_k = block.intrinsics.to(device)
        noise = torch.randn_like(target)
        sigma = shifted_sigma(
            torch.rand(
                target.shape[0],
                device=device,
                dtype=torch.float32,
            ),
            config.timestep_shift,
        )
        sigma_broadcast = sigma.view(-1, 1, 1, 1, 1).to(target.dtype)
        noisy = (1.0 - sigma_broadcast) * target + sigma_broadcast * noise
        velocity_target = noise - target

        teacher_features, teacher_grid = teacher.encode(
            block.geometry_query_rgb,
            item_name=batch.sample.item_name,
            frame_index=block.geometry_query_index,
        )
        predicted, geometry_prediction, memory = model(
            noisy,
            sigma * 1000.0,
            prompt_embeds,
            history_latents=selected_latents,
            history_c2w=selected_c2w,
            history_intrinsics=selected_k,
            target_c2w=target_c2w,
            target_intrinsics=target_k,
            query_c2w=block.geometry_query_c2w.to(device),
            query_intrinsics=block.geometry_query_intrinsics.to(device),
            teacher_image_hw=(
                teacher_grid[0] * teacher.config.patch_size,
                teacher_grid[1] * teacher.config.patch_size,
            ),
            encoder_attention_mask=prompt_mask,
        )
        flow_loss = F.mse_loss(
            predicted.float(),
            velocity_target.float(),
        )
        if geometry_prediction.shape[1] != teacher_features.shape[1]:
            raise RuntimeError(
                "VGGT patch count does not match the geometry head: "
                f"{teacher_features.shape[1]} vs "
                f"{geometry_prediction.shape[1]}"
            )
        geometry_loss = (
            1.0
            - F.cosine_similarity(
                geometry_prediction.float(),
                teacher_features.to(
                    device=geometry_prediction.device,
                    dtype=torch.float32,
                ),
                dim=-1,
                eps=1e-8,
            ).mean()
        )
        loss = flow_loss + config.geometry_loss_weight * geometry_loss

        # A write is useful only when another target block follows and can
        # train against the updated memory.  The default one-block schedule
        # therefore performs no self-forcing or teacher-forced memory update.
        has_next_block = block_index + 1 < len(batch.query_blocks)
        use_prediction = False
        if has_next_block:
            use_prediction = (
                config.predicted_update_probability > 0
                and float(torch.rand((), device=device).item())
                < config.predicted_update_probability
            )
            if use_prediction:
                update_latents = noisy - sigma_broadcast * predicted
            else:
                update_latents = target
            update_latents = update_latents.detach().cpu()
            latent_time_step = (
                batch.sample.capture_frame_stride
                * config.vae_temporal_stride
            )
            next_time = int(history_times.max().item()) + latent_time_step
            update_times = (
                torch.arange(
                    update_latents.shape[2],
                    dtype=history_times.dtype,
                )
                * latent_time_step
                + next_time
            )
            history_latents = torch.cat(
                (history_latents, update_latents),
                dim=2,
            )
            history_c2w = torch.cat(
                (history_c2w, block.c2w.cpu()),
                dim=1,
            )
            history_intrinsics = torch.cat(
                (history_intrinsics, block.intrinsics.cpu()),
                dim=1,
            )
            history_times = torch.cat((history_times, update_times), dim=0)

        if backward is None:
            losses.append(loss)
        else:
            # Backprop each query block immediately so two 81-frame LingBot
            # graphs never coexist. Scaling keeps the scene objective equal
            # to the mean over blocks.
            backward(loss / len(batch.query_blocks))
            losses.append(loss.detach())
        flow_losses.append(flow_loss.detach())
        geometry_losses.append(geometry_loss.detach())
        sigmas.append(sigma.mean().detach())
        memory_norms.append(
            memory.detach().float().norm(dim=-1).mean()
        )
        candidate_counts.append(
            torch.tensor(
                float(history_candidates_before),
                device=device,
            )
        )
        retained_counts.append(
            torch.tensor(float(selected.numel()), device=device)
        )
        predicted_updates.append(
            torch.tensor(float(use_prediction), device=device)
        )

    return {
        "loss": torch.stack(losses).mean(),
        "flow_loss": torch.stack(flow_losses).mean(),
        "geometry_loss": torch.stack(geometry_losses).mean(),
        "sigma": torch.stack(sigmas).mean(),
        "memory_norm": torch.stack(memory_norms).mean(),
        "history_candidates": torch.stack(candidate_counts).mean(),
        "history_retained": torch.stack(retained_counts).mean(),
        "predicted_update_fraction": torch.stack(predicted_updates).mean(),
        "capture_latent_frames": torch.tensor(
            float(batch.capture_latents.shape[2]),
            device=device,
        ),
        "final_history_latent_frames": torch.tensor(
            float(history_latents.shape[2]),
            device=device,
        ),
        "trajectory_overlap_score": torch.tensor(
            batch.sample.trajectory_overlap_score,
            device=device,
        ),
    }
