from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import random
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .geometry import (
    center_resize_crop,
    intrinsics_vector_to_matrix,
    normalize_c2w_to_first_capture,
    resize_crop_intrinsics,
)

logger = logging.getLogger(__name__)

# ViPE artifacts and the usable RGB stream are indexed 0,5,10,... in the
# source video.  The whole GIM pipeline operates on a contiguous 0..N-1
# internal timeline after this conversion.
SOURCE_FRAME_STRIDE = 5


def _is_rclone_remote(path: str) -> bool:
    if os.path.exists(path):
        return False
    head = path.split("/", 1)[0]
    return ":" in head and not path.startswith(("s3://", "http://", "https://"))


def _join_remote(root: str, child: str) -> str:
    return f"{root.rstrip('/')}/{child.strip('/')}"


def _rclone_environment(clear_proxy: bool) -> dict[str, str]:
    environment = dict(os.environ)
    if clear_proxy:
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "http_proxy",
            "https_proxy",
            "all_proxy",
        ):
            environment.pop(key, None)
    return environment


@dataclass(frozen=True)
class RcloneConfig:
    binary: str = "rclone"
    config_path: Optional[str] = None
    clear_proxy: bool = True
    transfers: int = 32
    checkers: int = 32

    def base_command(self) -> list[str]:
        command = [self.binary]
        if self.config_path:
            command.extend(["--config", self.config_path])
        return command


class RoomTourItemCache:
    """Materialize only RGB, pose, intrinsics, and metadata for one scene."""

    INCLUDE_PATTERNS = (
        "/RGB/*[05].jpg",
        "/RGB/*[05].jpeg",
        "/RGB/*[05].png",
        "/RGB/*[05].webp",
        "/RGB/*[05].JPG",
        "/RGB/*[05].JPEG",
        "/RGB/*[05].PNG",
        "/RGB/*[05].WEBP",
        "/chunk_metadata.json",
        "/vipe/vipe_artifacts/pose/video.npz",
        "/vipe/vipe_artifacts/intrinsics/video.npz",
        "/vipe/vipe_artifacts/intrinsics/video_camera.txt",
    )

    def __init__(
        self,
        dataset_root: str,
        cache_root: str | Path,
        *,
        rclone: Optional[RcloneConfig] = None,
    ) -> None:
        self.dataset_root = dataset_root.rstrip("/")
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.rclone = rclone or RcloneConfig()
        self.remote = _is_rclone_remote(dataset_root)

    def list_items(self) -> list[str]:
        if not self.remote:
            root = Path(self.dataset_root)
            items = sorted(path.name for path in root.iterdir() if path.is_dir())
            if not items:
                raise FileNotFoundError(f"no item directories below {root}")
            return items
        result = subprocess.run(
            [
                *self.rclone.base_command(),
                "lsf",
                self.dataset_root,
                "--dirs-only",
                "--max-depth",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=_rclone_environment(self.rclone.clear_proxy),
        )
        items = sorted(
            line.strip().rstrip("/")
            for line in result.stdout.splitlines()
            if line.strip()
        )
        if not items:
            raise RuntimeError(f"rclone found no items below {self.dataset_root}")
        return items

    @contextlib.contextmanager
    def _item_lock(self, item_name: str) -> Iterator[None]:
        lock_path = self.cache_root / f".{item_name}.gim.lock"
        with lock_path.open("a+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def materialize(self, item_name: str) -> Path:
        if not self.remote:
            path = Path(self.dataset_root) / item_name
            if not path.is_dir():
                raise FileNotFoundError(path)
            return path
        destination = self.cache_root / item_name
        marker_path = destination / ".gim_world_cache_complete.json"
        if marker_path.is_file():
            os.utime(marker_path, None)
            return destination
        with self._item_lock(item_name):
            if marker_path.is_file():
                os.utime(marker_path, None)
                return destination
            temporary = self.cache_root / f".partial-gim-{item_name}-{uuid.uuid4().hex}"
            temporary.mkdir(parents=True)
            command = [
                *self.rclone.base_command(),
                "copy",
                _join_remote(self.dataset_root, item_name),
                str(temporary),
                "--transfers",
                str(self.rclone.transfers),
                "--checkers",
                str(self.rclone.checkers),
                "--create-empty-src-dirs",
            ]
            # One ordered filter list avoids rclone's ambiguous
            # --include/--exclude warning.
            for pattern in self.INCLUDE_PATTERNS:
                command.extend(["--filter", f"+ {pattern}"])
            command.extend(["--filter", "- **"])
            try:
                subprocess.run(
                    command,
                    check=True,
                    env=_rclone_environment(self.rclone.clear_proxy),
                )
                marker = {
                    "source": _join_remote(self.dataset_root, item_name),
                    "item": item_name,
                    "source_frame_stride": SOURCE_FRAME_STRIDE,
                    "include_patterns": list(self.INCLUDE_PATTERNS),
                }
                (temporary / ".gim_world_cache_complete.json").write_text(
                    json.dumps(marker, indent=2) + "\n",
                    encoding="utf-8",
                )
                if destination.exists():
                    shutil.rmtree(destination)
                os.replace(temporary, destination)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise
        return destination


def _load_npz_mapping(path: Path) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    with np.load(path) as payload:
        for source_index, value in zip(
            payload["inds"],
            payload["data"],
            strict=True,
        ):
            source_index = int(source_index)
            if source_index % SOURCE_FRAME_STRIDE:
                continue
            output[source_index // SOURCE_FRAME_STRIDE] = np.asarray(value)
    if not output:
        raise ValueError(
            f"{path} has no indices divisible by {SOURCE_FRAME_STRIDE}"
        )
    return output


@dataclass(frozen=True)
class GeometryMemorySampleConfig:
    height: int = 480
    width: int = 832
    target_rgb_frames: int = 81
    vae_temporal_stride: int = 4
    min_memory_rgb_frames: int = 800
    target_guard_rgb_frames: int = 128
    samples_per_item: int = 16
    context_policy: str = "prefix"

    def validate(self) -> None:
        if self.height % 16 or self.width % 16:
            raise ValueError("height and width must be multiples of 16")
        if self.target_rgb_frames < 2:
            raise ValueError("target_rgb_frames must be at least 2")
        if (self.target_rgb_frames - 1) % self.vae_temporal_stride:
            raise ValueError(
                "target_rgb_frames must equal 1 + k * vae_temporal_stride"
            )
        if self.context_policy not in {"all_except_target", "prefix"}:
            raise ValueError("context_policy must be all_except_target or prefix")
        if (
            self.context_policy == "all_except_target"
            and self.target_guard_rgb_frames < 128
        ):
            raise ValueError(
                "all_except_target requires target_guard_rgb_frames >= 128 "
                "to prevent a future causal Wan latent from containing target "
                "RGB; use context_policy=prefix for the paper protocol"
            )

    @property
    def target_latent_frames(self) -> int:
        return 1 + (self.target_rgb_frames - 1) // self.vae_temporal_stride


@dataclass(frozen=True)
class RoomTourSample:
    item_name: str
    local_root: str
    target_start: int
    target_rgb_indices: tuple[int, ...]
    memory_rgb_indices: tuple[int, ...]
    geometry_query_index: int
    image_hw: tuple[int, int]


class VipeRoomTourItem:
    """Indexed view of a materialized sparse ViPE room tour."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        metadata_path = self.root / "chunk_metadata.json"
        self.metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        artifact_root = self.root / "vipe" / "vipe_artifacts"
        self.pose_by_index = _load_npz_mapping(
            artifact_root / "pose" / "video.npz"
        )
        self.intrinsics_by_index = _load_npz_mapping(
            artifact_root / "intrinsics" / "video.npz"
        )
        self._validate_camera_type(
            artifact_root / "intrinsics" / "video_camera.txt"
        )
        self.rgb_by_index = self._index_rgb()
        self.indices = sorted(
            set(self.rgb_by_index)
            & set(self.pose_by_index)
            & set(self.intrinsics_by_index)
        )
        if not self.indices:
            raise RuntimeError(f"{self.root.name} has no common RGB/camera frames")
        expected = list(range(self.indices[0], self.indices[-1] + 1))
        if self.indices != expected:
            missing = sorted(set(expected) - set(self.indices))
            raise RuntimeError(
                f"{self.root.name} common timeline is not contiguous; "
                f"first missing indices: {missing[:16]}"
            )

    def _validate_camera_type(self, path: Path) -> None:
        if not path.is_file():
            return
        unsupported = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                camera_type = line.split(":", 1)[1].strip().upper()
                if camera_type != "PINHOLE":
                    unsupported.add(camera_type)
        if unsupported:
            raise NotImplementedError(
                f"only PINHOLE cameras are supported, found {sorted(unsupported)}"
            )

    def _index_rgb(self) -> dict[int, Path]:
        output: dict[int, Path] = {}
        rgb_root = self.root / "RGB"
        for path in sorted(rgb_root.glob("*")):
            if (
                not path.is_file()
                or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
            ):
                continue
            try:
                source_index = int(path.stem)
            except ValueError:
                continue
            if source_index % SOURCE_FRAME_STRIDE:
                continue
            output[source_index // SOURCE_FRAME_STRIDE] = path
        if not output:
            raise FileNotFoundError(
                f"no RGB frames divisible by {SOURCE_FRAME_STRIDE} below {rgb_root}"
            )
        return output

    @property
    def source_hw(self) -> tuple[int, int]:
        if "height" in self.metadata and "width" in self.metadata:
            return int(self.metadata["height"]), int(self.metadata["width"])
        with Image.open(self.rgb_by_index[self.indices[0]]) as image:
            return image.height, image.width

    def read_rgb(
        self,
        index: int,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        with Image.open(self.rgb_by_index[index]) as image:
            image = image.convert("RGB")
            transform = center_resize_crop(
                (image.height, image.width),
                target_hw,
            )
            image = image.resize(
                (transform.resized_width, transform.resized_height),
                resample=Image.Resampling.BICUBIC,
            )
            left = int(transform.crop_left)
            top = int(transform.crop_top)
            image = image.crop(
                (
                    left,
                    top,
                    left + transform.target_width,
                    top + transform.target_height,
                )
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def read_video(
        self,
        indices: list[int] | tuple[int, ...],
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        return torch.stack(
            [self.read_rgb(index, target_hw) for index in indices],
            dim=1,
        )

    def cameras(
        self,
        indices: list[int] | tuple[int, ...] | torch.Tensor,
        target_hw: tuple[int, int],
        *,
        origin_index: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(indices, torch.Tensor):
            index_list = [int(value) for value in indices.tolist()]
        else:
            index_list = [int(value) for value in indices]
        raw_pose = torch.from_numpy(
            np.stack([self.pose_by_index[index] for index in index_list])
        ).float()
        origin = self.indices[0] if origin_index is None else int(origin_index)
        first_pose = torch.from_numpy(self.pose_by_index[origin]).float()
        pose = normalize_c2w_to_first_capture(raw_pose, first_pose)
        intrinsics_vector = torch.from_numpy(
            np.stack([self.intrinsics_by_index[index] for index in index_list])
        ).float()
        intrinsics = intrinsics_vector_to_matrix(
            resize_crop_intrinsics(
                intrinsics_vector,
                self.source_hw,
                target_hw,
            )
        )
        return pose, intrinsics

    def valid_target_starts(
        self,
        config: GeometryMemorySampleConfig,
    ) -> list[int]:
        config.validate()
        first, last = self.indices[0], self.indices[-1]
        starts: list[int] = []
        for start in range(first, last - config.target_rgb_frames + 2):
            target_end = start + config.target_rgb_frames - 1
            if config.context_policy == "prefix":
                memory_count = start - first
            else:
                excluded_start = max(
                    first,
                    start - config.target_guard_rgb_frames,
                )
                excluded_end = min(
                    last,
                    target_end + config.target_guard_rgb_frames,
                )
                memory_count = len(self.indices) - (
                    excluded_end - excluded_start + 1
                )
            if memory_count >= config.min_memory_rgb_frames:
                starts.append(start)
        return starts

    def make_sample(
        self,
        config: GeometryMemorySampleConfig,
        rng: random.Random,
        *,
        target_start: Optional[int] = None,
    ) -> RoomTourSample:
        starts = self.valid_target_starts(config)
        if not starts:
            raise RuntimeError(
                f"{self.root.name} has no valid GIM-World training window"
            )
        if target_start is None:
            target_start = rng.choice(starts)
        elif target_start not in starts:
            raise ValueError(
                f"target_start={target_start} is invalid; valid range begins "
                f"{starts[:4]} and ends {starts[-4:]}"
            )
        target = tuple(
            range(target_start, target_start + config.target_rgb_frames)
        )
        if config.context_policy == "prefix":
            memory = tuple(
                index for index in self.indices if index < target_start
            )
        else:
            excluded_start = target_start - config.target_guard_rgb_frames
            excluded_end = target[-1] + config.target_guard_rgb_frames
            memory = tuple(
                index
                for index in self.indices
                if not excluded_start <= index <= excluded_end
            )
        query_index = rng.choice(memory)
        return RoomTourSample(
            item_name=self.root.name,
            local_root=str(self.root),
            target_start=target_start,
            target_rgb_indices=target,
            memory_rgb_indices=memory,
            geometry_query_index=query_index,
            image_hw=(config.height, config.width),
        )


class RemoteVipeRoomTourDataset(IterableDataset):
    """Scene-reuse stream; samples carry indices, not thousand-frame tensors."""

    def __init__(
        self,
        dataset_root: str,
        cache_root: str | Path,
        sample_config: GeometryMemorySampleConfig,
        *,
        rclone: Optional[RcloneConfig] = None,
        item_list: Optional[list[str]] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.item_cache = RoomTourItemCache(
            dataset_root,
            cache_root,
            rclone=rclone,
        )
        self.sample_config = sample_config
        self.items = item_list or self.item_cache.list_items()
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[RoomTourSample]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        shard_id = self.rank * worker_count + worker_id
        shard_count = self.world_size * worker_count
        items = self.items[shard_id::shard_count]
        if not items:
            raise RuntimeError(
                f"dataset shard {shard_id}/{shard_count} has no items"
            )
        rng = random.Random(self.seed + 10_007 * shard_id)
        while True:
            shuffled = list(items)
            rng.shuffle(shuffled)
            for item_name in shuffled:
                try:
                    item = VipeRoomTourItem(
                        self.item_cache.materialize(item_name)
                    )
                    for _ in range(self.sample_config.samples_per_item):
                        yield item.make_sample(self.sample_config, rng)
                except Exception:
                    logger.exception(
                        "failed to prepare/sample room-tour item %s",
                        item_name,
                    )
