from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from .data import RoomTourSample, VipeRoomTourItem
from .model import GIMWorldLingBotModel
from .profiling import ProfileTimings, profiling_scope, synchronized_stage
from .pruning import MIGreedyPruner
from .teacher import VGGTGeometryTeacher


@dataclass(frozen=True)
class GIMTrainingConfig:
    pruning_budget: int = 200
    geometry_loss_weight: float = 0.05
    vae_encode_chunk_rgb_frames: int = 4
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


def _require_independent_wan_vae(vae: torch.nn.Module) -> None:
    missing = [name for name in ("encode",) if not hasattr(vae, name)]
    if missing:
        raise TypeError(
            "independent-frame encoding requires a compatible diffusers "
            "AutoencoderKLWan; "
            f"missing attributes: {missing}"
        )
    if getattr(vae.config, "patch_size", None) is not None:
        raise NotImplementedError(
            "independent-frame encoding does not support a patchified Wan VAE"
        )


@torch.no_grad()
def encode_wan_frames_independently(
    vae: torch.nn.Module,
    item: VipeRoomTourItem,
    indices: list[int] | tuple[int, ...],
    target_hw: tuple[int, int],
    *,
    read_chunk_rgb_frames: int,
    device: torch.device,
    dtype: torch.dtype,
    profile_timings: ProfileTimings | None = None,
    profile_prefix: str = "vae",
) -> torch.Tensor:
    """Encode every RGB frame as an independent one-frame Wan sample.

    Frames share only the VAE batch dimension. They never share causal feature
    state, so the output temporal axis has exactly one latent per input RGB and
    can be paired one-to-one with camera poses.
    """

    _require_independent_wan_vae(vae)
    indices = [int(index) for index in indices]
    if not indices:
        raise ValueError("cannot encode an empty frame set")
    if read_chunk_rgb_frames < 1:
        raise ValueError("read_chunk_rgb_frames must be positive")
    autocast = (
        torch.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        else contextlib.nullcontext()
    )
    latent_chunks: list[torch.Tensor] = []
    for position in range(0, len(indices), read_chunk_rgb_frames):
        read_indices = indices[position : position + read_chunk_rgb_frames]
        with synchronized_stage(
            profile_timings,
            f"{profile_prefix}_rgb_read",
            device,
        ):
            video = item.read_video(read_indices, target_hw)
        with synchronized_stage(
            profile_timings,
            f"{profile_prefix}_h2d_preprocess",
            device,
        ):
            images = (
                video.permute(1, 0, 2, 3)
                .unsqueeze(2)
                .to(device=device, dtype=torch.float32)
                .mul(2.0)
                .sub(1.0)
            )
        with synchronized_stage(
            profile_timings,
            f"{profile_prefix}_vae_forward",
            device,
        ):
            with autocast:
                encoded = vae.encode(images)
            if hasattr(encoded, "latent_dist"):
                distribution = encoded.latent_dist
                latents = (
                    distribution.mode()
                    if callable(getattr(distribution, "mode", None))
                    else distribution.mean
                )
            elif isinstance(encoded, tuple):
                latents = encoded[0]
            else:
                latents = encoded
            if latents.ndim != 5 or latents.shape[2] != 1:
                raise RuntimeError(
                    "one-frame Wan encode must return [N,C,1,H,W], got "
                    f"{tuple(latents.shape)}"
                )
            dit_latents = _vae_latent_to_dit(vae, latents)
        with synchronized_stage(
            profile_timings,
            f"{profile_prefix}_latent_d2h",
            device,
        ):
            latent_chunks.append(
                dit_latents.squeeze(2).permute(1, 0, 2, 3).unsqueeze(0).cpu()
            )
        del video, images, encoded, latents, dit_latents
    output = torch.cat(latent_chunks, dim=2)
    if output.shape[2] != len(indices):
        raise RuntimeError(
            f"Wan returned {output.shape[2]} latents for {len(indices)} frames"
        )
    return output


@torch.no_grad()
def prepare_gim_trajectory_online(
    sample: RoomTourSample,
    *,
    vae: torch.nn.Module,
    vae_encode_chunk_rgb_frames: int,
    device: torch.device,
    compute_dtype: torch.dtype,
    profile_timings: ProfileTimings | None = None,
) -> PreparedTrajectoryBatch:
    """Read RGB and run every VAE encoding online for one scene iteration."""

    with synchronized_stage(
        profile_timings,
        "scene_index_init",
        device,
    ):
        item = VipeRoomTourItem(Path(sample.local_root))
    origin_index = sample.capture_start
    capture_latents = encode_wan_frames_independently(
        vae,
        item,
        sample.capture_rgb_indices,
        sample.image_hw,
        read_chunk_rgb_frames=vae_encode_chunk_rgb_frames,
        device=device,
        dtype=compute_dtype,
        profile_timings=profile_timings,
        profile_prefix="memory",
    )
    capture_latent_indices = sample.capture_rgb_indices
    with synchronized_stage(
        profile_timings,
        "memory_camera_metadata",
        device,
    ):
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
        latents = encode_wan_frames_independently(
            vae,
            item,
            rgb_indices,
            sample.image_hw,
            read_chunk_rgb_frames=vae_encode_chunk_rgb_frames,
            device=device,
            dtype=compute_dtype,
            profile_timings=profile_timings,
            profile_prefix="target",
        )
        latent_rgb_indices = rgb_indices
        with synchronized_stage(
            profile_timings,
            "target_camera_metadata",
            device,
        ):
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
        with synchronized_stage(
            profile_timings,
            "geometry_rgb_read",
            device,
        ):
            geometry_query_rgb = item.read_rgb(
                geometry_query_index,
                sample.image_hw,
            ).unsqueeze(0)
        query_blocks.append(
            PreparedQueryBlock(
                rgb_indices=rgb_indices,
                latent_rgb_indices=latent_rgb_indices,
                latents=latents,
                c2w=c2w.unsqueeze(0),
                intrinsics=intrinsics.unsqueeze(0),
                geometry_query_index=geometry_query_index,
                geometry_query_rgb=geometry_query_rgb,
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
    profile_timings: ProfileTimings | None = None,
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
        with synchronized_stage(
            profile_timings,
            "history_prune_transfer",
            device,
        ):
            selected_latents, selected_c2w, selected_k, selected = (
                _select_history(
                    history_latents,
                    history_c2w,
                    history_intrinsics,
                    history_times,
                    pruner=pruner,
                    budget=config.pruning_budget,
                    device=device,
                    dtype=compute_dtype,
                )
            )
        with synchronized_stage(profile_timings, "flow_setup", device):
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

        with synchronized_stage(profile_timings, "vggt_forward", device):
            teacher_features, teacher_grid = teacher.encode(
                block.geometry_query_rgb,
                item_name=batch.sample.item_name,
                frame_index=block.geometry_query_index,
            )
        # DDP may copy ordinary Python kwargs. A rank-local context therefore
        # carries the timing sink through the wrapper without changing any
        # tensor input or relying on mutation of a scattered dict.
        with profiling_scope(profile_timings):
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
        with synchronized_stage(profile_timings, "loss_forward", device):
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
        with synchronized_stage(
            profile_timings,
            "dynamic_memory_update",
            device,
        ):
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
                update_times = torch.tensor(
                    [
                        int(index) - batch.sample.capture_start
                        for index in block.latent_rgb_indices
                    ],
                    dtype=history_times.dtype,
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
            # Backprop each query block immediately so two target-chunk
            # LingBot graphs never coexist. Scaling keeps the scene objective
            # equal to the mean over blocks.
            with synchronized_stage(profile_timings, "backward", device):
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
        "retrieval_coverage_score": torch.tensor(
            batch.sample.retrieval_coverage_score,
            device=device,
        ),
    }
