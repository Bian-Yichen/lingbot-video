from __future__ import annotations

import contextlib
import fcntl
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

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


def _require_streamable_wan_vae(
    vae: torch.nn.Module,
    temporal_stride: int,
) -> None:
    missing = [
        name
        for name in (
            "clear_cache",
            "encoder",
            "quant_conv",
            "_enc_feat_map",
            "_enc_conv_idx",
        )
        if not hasattr(vae, name)
    ]
    if missing:
        raise TypeError(
            "scene caching requires diffusers AutoencoderKLWan's official "
            f"feature-cache API; missing attributes: {missing}"
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
            "streaming cache does not support a patchified Wan VAE"
        )


@torch.no_grad()
def encode_wan_scene_streaming(
    vae: torch.nn.Module,
    item: VipeRoomTourItem,
    indices: list[int],
    target_hw: tuple[int, int],
    *,
    temporal_stride: int,
    read_chunk_rgb_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the exact AutoencoderKLWan causal feature-cache schedule.

    AutoencoderKLWan._encode feeds the first RGB frame alone and then groups
    of four frames through one persistent cache for every causal convolution.
    Calling ``vae.encode`` independently on overlapping clips resets all those
    caches and is therefore not equivalent.  This function preserves the
    official schedule while keeping only one read chunk and one 1/4-frame
    encoder group on the accelerator.
    """

    _require_streamable_wan_vae(vae, temporal_stride)
    if not indices:
        raise ValueError("cannot encode an empty scene")

    autocast = (
        torch.autocast("cuda", dtype=dtype)
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        else contextlib.nullcontext()
    )
    latent_chunks: list[torch.Tensor] = []
    vae.clear_cache()
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
                group = group.unsqueeze(0).to(
                    device=device,
                    dtype=torch.float32,
                )
                group = group.mul(2.0).sub(1.0)
                vae._enc_conv_idx = [0]
                with autocast:
                    features = vae.encoder(
                        group,
                        feat_cache=vae._enc_feat_map,
                        feat_idx=vae._enc_conv_idx,
                    )
                    moments = vae.quant_conv(features)
                # DiagonalGaussianDistribution.mode() is exactly its mean,
                # i.e. the first half of the moment channels.
                latents = moments.chunk(2, dim=1)[0]
                latent_chunks.append(
                    _vae_latent_to_dit(vae, latents).cpu()
                )
                local_position += group.shape[2]
                del group, features, moments, latents
            position += read_count
            first_read = False
            del video
    finally:
        vae.clear_cache()

    return torch.cat(latent_chunks, dim=2)


class SceneLatentCache:
    """Encode a materialized scene once, then reuse it for many iterations."""

    def __init__(
        self,
        cache_root: str | Path,
        *,
        height: int,
        width: int,
        temporal_stride: int,
        chunk_rgb_frames: int,
        storage_dtype: torch.dtype = torch.float16,
        max_in_memory_scenes: int = 2,
    ) -> None:
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.height = int(height)
        self.width = int(width)
        self.temporal_stride = int(temporal_stride)
        self.chunk_rgb_frames = int(chunk_rgb_frames)
        self.storage_dtype = storage_dtype
        self.max_in_memory_scenes = int(max_in_memory_scenes)
        if (
            self.chunk_rgb_frames <= self.temporal_stride
            or (self.chunk_rgb_frames - 1) % self.temporal_stride
        ):
            raise ValueError(
                "chunk_rgb_frames must equal 1 + k * temporal_stride "
                "with k >= 1"
            )
        if self.max_in_memory_scenes < 1:
            raise ValueError("max_in_memory_scenes must be positive")
        self._memory: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()

    def _path(self, item: VipeRoomTourItem) -> Path:
        return (
            self.cache_root
            / item.root.name
            / f"lingbot_vae_{self.height}x{self.width}_s{self.temporal_stride}.pt"
        )

    def _target_path(
        self,
        item: VipeRoomTourItem,
        target_indices: list[int] | tuple[int, ...],
    ) -> Path:
        return (
            self.cache_root
            / item.root.name
            / "target_clips"
            / (
                f"{int(target_indices[0]):06d}_{int(target_indices[-1]):06d}"
                f"_{self.height}x{self.width}_s{self.temporal_stride}.pt"
            )
        )

    @contextlib.contextmanager
    def _lock(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _valid_payload(
        self,
        payload: dict,
        item: VipeRoomTourItem,
    ) -> bool:
        metadata = payload.get("metadata", {})
        return (
            metadata.get("item_name") == item.root.name
            and metadata.get("height") == self.height
            and metadata.get("width") == self.width
            and metadata.get("temporal_stride") == self.temporal_stride
            and metadata.get("rgb_first") == item.indices[0]
            and metadata.get("rgb_last") == item.indices[-1]
            and metadata.get("rgb_count") == len(item.indices)
            and metadata.get("encoder_mode")
            == "wan_official_feature_cache_v1"
        )

    @torch.no_grad()
    def _build(
        self,
        item: VipeRoomTourItem,
        vae: torch.nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor | dict]:
        all_indices = item.indices
        latents = encode_wan_scene_streaming(
            vae,
            item,
            all_indices,
            (self.height, self.width),
            temporal_stride=self.temporal_stride,
            read_chunk_rgb_frames=self.chunk_rgb_frames,
            device=device,
            dtype=dtype,
        ).squeeze(0)
        latent_rgb_indices = torch.tensor(
            all_indices[:: self.temporal_stride],
            dtype=torch.long,
        )
        if latents.shape[1] != latent_rgb_indices.numel():
            raise RuntimeError(
                "chunked VAE temporal alignment failed: "
                f"{latents.shape[1]} latents for "
                f"{latent_rgb_indices.numel()} expected indices"
            )
        return {
            "latents": latents.to(self.storage_dtype).contiguous(),
            "rgb_indices": latent_rgb_indices,
            "metadata": {
                "item_name": item.root.name,
                "height": self.height,
                "width": self.width,
                "temporal_stride": self.temporal_stride,
                "rgb_first": item.indices[0],
                "rgb_last": item.indices[-1],
                "rgb_count": len(item.indices),
                "chunk_rgb_frames": self.chunk_rgb_frames,
                "encoder_mode": "wan_official_feature_cache_v1",
            },
        }

    def get(
        self,
        item: VipeRoomTourItem,
        vae: torch.nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        key = str(self._path(item))
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        path = self._path(item)
        with self._lock(path):
            payload = None
            if path.is_file():
                candidate = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                )
                if self._valid_payload(candidate, item):
                    payload = candidate
            if payload is None:
                payload = self._build(
                    item,
                    vae,
                    device=device,
                    dtype=dtype,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=path.parent,
                    suffix=".pt",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                try:
                    torch.save(payload, temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        result = {
            "latents": payload["latents"],
            "rgb_indices": payload["rgb_indices"],
        }
        self._memory[key] = result
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_in_memory_scenes:
            self._memory.popitem(last=False)
        return result

    @torch.no_grad()
    def get_target(
        self,
        item: VipeRoomTourItem,
        target_indices: list[int] | tuple[int, ...],
        vae: torch.nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode one training target with a fresh Wan causal state.

        A slice of the full-scene latent stream is not equivalent: only the
        first latent of an independently decoded clip has first-chunk
        semantics.  Target clips therefore have a separate content cache.
        """

        if not target_indices:
            raise ValueError("target_indices cannot be empty")
        path = self._target_path(item, target_indices)
        with self._lock(path):
            payload = None
            if path.is_file():
                candidate = torch.load(
                    path,
                    map_location="cpu",
                    weights_only=True,
                )
                metadata = candidate.get("metadata", {})
                if (
                    metadata.get("item_name") == item.root.name
                    and metadata.get("indices")
                    == [int(index) for index in target_indices]
                    and metadata.get("height") == self.height
                    and metadata.get("width") == self.width
                    and metadata.get("temporal_stride")
                    == self.temporal_stride
                    and metadata.get("encoder_mode")
                    == "wan_official_feature_cache_v1_fresh_clip"
                ):
                    payload = candidate
            if payload is None:
                latent = encode_wan_scene_streaming(
                    vae,
                    item,
                    [int(index) for index in target_indices],
                    (self.height, self.width),
                    temporal_stride=self.temporal_stride,
                    read_chunk_rgb_frames=self.chunk_rgb_frames,
                    device=device,
                    dtype=dtype,
                ).squeeze(0)
                expected = 1 + (
                    len(target_indices) - 1
                ) // self.temporal_stride
                if latent.shape[1] != expected:
                    raise RuntimeError(
                        "target VAE temporal alignment failed: "
                        f"{latent.shape[1]} latents for {expected} expected"
                    )
                payload = {
                    "latents": latent.to(self.storage_dtype).contiguous(),
                    "metadata": {
                        "item_name": item.root.name,
                        "indices": [int(index) for index in target_indices],
                        "height": self.height,
                        "width": self.width,
                        "temporal_stride": self.temporal_stride,
                        "encoder_mode": (
                            "wan_official_feature_cache_v1_fresh_clip"
                        ),
                    },
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    dir=path.parent,
                    suffix=".pt",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                try:
                    torch.save(payload, temporary)
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
        return payload["latents"]


def shifted_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def prepare_gim_batch(
    sample: RoomTourSample,
    *,
    vae: torch.nn.Module,
    latent_cache: SceneLatentCache,
    pruner: MIGreedyPruner,
    pruning_budget: int,
    vae_temporal_stride: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor | VipeRoomTourItem]:
    item = VipeRoomTourItem(Path(sample.local_root))
    cached = latent_cache.get(
        item,
        vae,
        device=device,
        dtype=compute_dtype,
    )
    scene_latents = cached["latents"]
    latent_rgb_indices = cached["rgb_indices"]
    latent_memory_set = set(sample.memory_rgb_indices)
    history_positions = torch.tensor(
        [
            position
            for position, rgb_index in enumerate(latent_rgb_indices.tolist())
            if int(rgb_index) in latent_memory_set
        ],
        dtype=torch.long,
    )
    if history_positions.numel() == 0:
        raise RuntimeError("sample has no history frames on the latent timeline")
    history_rgb_indices = latent_rgb_indices[history_positions]
    history_c2w_cpu, history_k_cpu = item.cameras(
        history_rgb_indices,
        sample.image_hw,
    )
    chosen_local = pruner.select(
        history_c2w_cpu,
        history_rgb_indices.float(),
        pruning_budget,
    )
    # The subset is a set in equation (15); restore chronological order before
    # applying temporal positional encoding.
    chosen_local = chosen_local.sort().values.cpu()
    chosen_positions = history_positions[chosen_local]
    retained_rgb_indices = latent_rgb_indices[chosen_positions]

    target_latents = latent_cache.get_target(
        item,
        sample.target_rgb_indices,
        vae,
        device=device,
        dtype=compute_dtype,
    )
    target_latent_rgb = tuple(
        sample.target_rgb_indices[::vae_temporal_stride]
    )
    target_c2w, target_k = item.cameras(
        target_latent_rgb,
        sample.image_hw,
    )
    retained_c2w, retained_k = item.cameras(
        retained_rgb_indices,
        sample.image_hw,
    )
    query_c2w, query_k = item.cameras(
        [sample.geometry_query_index],
        sample.image_hw,
    )
    query_rgb = item.read_rgb(
        sample.geometry_query_index,
        sample.image_hw,
    ).unsqueeze(0)

    return {
        "item": item,
        "all_history_latents": scene_latents[:, history_positions].unsqueeze(0),
        "all_history_c2w": history_c2w_cpu.unsqueeze(0),
        "all_history_intrinsics": history_k_cpu.unsqueeze(0),
        "all_history_times": history_rgb_indices.clone(),
        "history_latents": scene_latents[:, chosen_positions]
        .unsqueeze(0)
        .to(device=device, dtype=compute_dtype),
        "history_c2w": retained_c2w.unsqueeze(0).to(device),
        "history_intrinsics": retained_k.unsqueeze(0).to(device),
        "target_latents": target_latents.unsqueeze(0).to(
            device=device,
            dtype=compute_dtype,
        ),
        "target_c2w": target_c2w.unsqueeze(0).to(device),
        "target_intrinsics": target_k.unsqueeze(0).to(device),
        "query_rgb": query_rgb,
        "query_c2w": query_c2w.unsqueeze(0).to(device),
        "query_intrinsics": query_k.unsqueeze(0).to(device),
        "history_candidates": torch.tensor(
            history_positions.numel(),
            device=device,
        ),
        "history_retained": torch.tensor(
            chosen_positions.numel(),
            device=device,
        ),
        "retained_rgb_indices": retained_rgb_indices,
    }


def gim_training_step(
    model: GIMWorldLingBotModel,
    batch: dict[str, torch.Tensor | VipeRoomTourItem],
    *,
    teacher: VGGTGeometryTeacher,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    config: GIMTrainingConfig,
    item_name: str,
    geometry_query_index: int,
) -> dict[str, torch.Tensor]:
    target = batch["target_latents"]
    noise = torch.randn_like(target)
    sigma = shifted_sigma(
        torch.rand(
            target.shape[0],
            device=target.device,
            dtype=torch.float32,
        ),
        config.timestep_shift,
    )
    sigma_broadcast = sigma.view(-1, 1, 1, 1, 1).to(target.dtype)
    noisy = (1.0 - sigma_broadcast) * target + sigma_broadcast * noise
    velocity_target = noise - target
    teacher_features, teacher_grid = teacher.encode(
        batch["query_rgb"],
        item_name=item_name,
        frame_index=geometry_query_index,
    )
    predicted, geometry_prediction, memory = model(
        noisy,
        sigma * 1000.0,
        prompt_embeds,
        history_latents=batch["history_latents"],
        history_c2w=batch["history_c2w"],
        history_intrinsics=batch["history_intrinsics"],
        target_c2w=batch["target_c2w"],
        target_intrinsics=batch["target_intrinsics"],
        query_c2w=batch["query_c2w"],
        query_intrinsics=batch["query_intrinsics"],
        teacher_image_hw=(
            teacher_grid[0] * teacher.config.patch_size,
            teacher_grid[1] * teacher.config.patch_size,
        ),
        encoder_attention_mask=prompt_mask,
    )
    flow_loss = F.mse_loss(predicted.float(), velocity_target.float())
    if geometry_prediction.shape[1] != teacher_features.shape[1]:
        raise RuntimeError(
            "VGGT patch count does not match the configured geometry head: "
            f"{teacher_features.shape[1]} vs {geometry_prediction.shape[1]}"
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
    return {
        "loss": loss,
        "flow_loss": flow_loss.detach(),
        "geometry_loss": geometry_loss.detach(),
        "sigma": sigma.mean().detach(),
        "memory_norm": memory.float().norm(dim=-1).mean().detach(),
        "history_candidates": batch["history_candidates"].float(),
        "history_retained": batch["history_retained"].float(),
    }
