from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import uuid
import zipfile
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
    resize_crop_tensor,
)

logger = logging.getLogger(__name__)


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


@dataclass
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
    """Node-local cache that downloads one useful item subset and reuses it."""

    INCLUDE_PATTERNS = (
        "/RGB/**",
        "/chunk_metadata.json",
        "/vipe/vipe_artifacts/pose/video.npz",
        "/vipe/vipe_artifacts/intrinsics/video.npz",
        "/vipe/vipe_artifacts/intrinsics/video_camera.txt",
        "/vipe/vipe_artifacts/depth/video.zip",
        "/vipe/vipe_artifacts/depth/video.zip.parts/*.zip",
        "/vipe/vipe_artifacts/depth/video.zip.parts/manifest.json",
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
        command = [
            *self.rclone.base_command(),
            "lsf",
            self.dataset_root,
            "--dirs-only",
            "--max-depth",
            "1",
        ]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            env=_rclone_environment(self.rclone.clear_proxy),
        )
        items = sorted(line.strip().rstrip("/") for line in result.stdout.splitlines() if line.strip())
        if not items:
            raise RuntimeError(f"rclone found no items below {self.dataset_root}")
        return items

    @contextlib.contextmanager
    def _item_lock(self, item_name: str) -> Iterator[None]:
        lock_path = self.cache_root / f".{item_name}.lock"
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
        complete = destination / ".latent_memory_cache_complete.json"
        if complete.is_file():
            os.utime(complete, None)
            return destination
        with self._item_lock(item_name):
            if complete.is_file():
                os.utime(complete, None)
                return destination
            temporary = self.cache_root / f".partial-{item_name}-{uuid.uuid4().hex}"
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
            for pattern in self.INCLUDE_PATTERNS:
                command.extend(["--include", pattern])
            command.extend(["--exclude", "*"])
            try:
                subprocess.run(
                    command,
                    check=True,
                    env=_rclone_environment(self.rclone.clear_proxy),
                )
                marker = {
                    "source": _join_remote(self.dataset_root, item_name),
                    "item": item_name,
                    "include_patterns": list(self.INCLUDE_PATTERNS),
                }
                (temporary / ".latent_memory_cache_complete.json").write_text(
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


@dataclass(frozen=True)
class LongTrajectorySampleConfig:
    height: int = 480
    width: int = 832
    capture_frames: int = 48
    history_min_frames: int = 256
    history_max_frames: int = 4096
    rollout_chunks: int = 2
    latent_frames_per_chunk: int = 9
    vae_temporal_stride: int = 4
    samples_per_item: int = 32
    min_depth: float = 0.1
    max_depth: float = 20.0
    max_sample_attempts: int = 64

    @property
    def total_latent_frames(self) -> int:
        return 1 + self.rollout_chunks * (self.latent_frames_per_chunk - 1)

    @property
    def target_rgb_frames(self) -> int:
        return 1 + (self.total_latent_frames - 1) * self.vae_temporal_stride


def _load_npz_mapping(path: Path) -> dict[int, np.ndarray]:
    with np.load(path) as payload:
        return {
            int(index): np.asarray(value)
            for index, value in zip(payload["inds"], payload["data"], strict=True)
        }


def _decode_exr(payload: bytes) -> np.ndarray:
    """Decode the Z channel written by ViPE, preferring its OpenEXR dependency."""

    with tempfile.NamedTemporaryFile(suffix=".exr") as temporary:
        temporary.write(payload)
        temporary.flush()
        try:
            import Imath
            import OpenEXR

            file = OpenEXR.InputFile(temporary.name)
            try:
                window = file.header()["dataWindow"]
                width = int(window.max.x - window.min.x + 1)
                height = int(window.max.y - window.min.y + 1)
                raw = file.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
                return np.frombuffer(raw, dtype=np.float32).reshape(height, width).copy()
            finally:
                file.close()
        except ImportError:
            try:
                import imageio.v3 as iio

                depth = iio.imread(temporary.name)
            except Exception as exc:
                raise RuntimeError(
                    "Depth EXR decoding needs OpenEXR+Imath (the same packages used by ViPE) "
                    "or an imageio backend with EXR support."
                ) from exc
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32, copy=False)


class VipeRoomTourItem:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.metadata = self._load_metadata()
        artifact_root = self.root / "vipe" / "vipe_artifacts"
        self.pose_by_index = _load_npz_mapping(artifact_root / "pose" / "video.npz")
        self.intrinsics_by_index = _load_npz_mapping(
            artifact_root / "intrinsics" / "video.npz"
        )
        self._validate_camera_type(artifact_root / "intrinsics" / "video_camera.txt")
        self.rgb_by_index = self._index_rgb()
        self.depth_location = self._index_depth(artifact_root / "depth")
        self._depth_cache: dict[int, np.ndarray] = {}
        self._depth_cache_order: list[int] = []
        self._depth_cache_limit = 128

    def _load_metadata(self) -> dict:
        path = self.root / "chunk_metadata.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

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
                f"only PINHOLE VIPE artifacts are supported, found {sorted(unsupported)}"
            )

    def _index_rgb(self) -> dict[int, Path]:
        output: dict[int, Path] = {}
        for path in sorted((self.root / "RGB").glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            try:
                output[int(path.stem)] = path
            except ValueError:
                continue
        if not output:
            raise FileNotFoundError(f"no indexed RGB frames below {self.root / 'RGB'}")
        return output

    @staticmethod
    def _index_depth(depth_root: Path) -> dict[int, tuple[Path, str]]:
        final_archive = depth_root / "video.zip"
        archives: list[Path] = []
        if final_archive.is_file():
            archives.append(final_archive)
        parts = depth_root / "video.zip.parts"
        if parts.is_dir():
            archives.extend(sorted(parts.glob("*.zip")))
        output: dict[int, tuple[Path, str]] = {}
        for archive_path in archives:
            try:
                with zipfile.ZipFile(archive_path, "r") as archive:
                    for member in archive.namelist():
                        try:
                            frame_index = int(Path(member).stem)
                        except ValueError:
                            continue
                        output.setdefault(frame_index, (archive_path, member))
            except zipfile.BadZipFile:
                logger.warning("ignoring incomplete depth archive %s", archive_path)
        if not output:
            raise FileNotFoundError(f"no complete depth frames below {depth_root}")
        return output

    @property
    def source_hw(self) -> tuple[int, int]:
        if "height" in self.metadata and "width" in self.metadata:
            return int(self.metadata["height"]), int(self.metadata["width"])
        first = self.rgb_by_index[min(self.rgb_by_index)]
        with Image.open(first) as image:
            width, height = image.size
        return height, width

    def read_rgb(self, index: int, target_hw: tuple[int, int]) -> torch.Tensor:
        path = self.rgb_by_index[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            actual_hw = (image.height, image.width)
            transform = center_resize_crop(actual_hw, target_hw)
            image = image.resize(
                (transform.resized_width, transform.resized_height),
                resample=Image.Resampling.BICUBIC,
            )
            left = int(transform.crop_left)
            top = int(transform.crop_top)
            image = image.crop(
                (left, top, left + transform.target_width, top + transform.target_height)
            )
            array = np.asarray(image, dtype=np.float32).copy() / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def read_depth(self, index: int, target_hw: tuple[int, int]) -> torch.Tensor:
        cached = self._depth_cache.get(index)
        if cached is None:
            archive_path, member = self.depth_location[index]
            with zipfile.ZipFile(archive_path, "r") as archive:
                cached = _decode_exr(archive.read(member))
            self._depth_cache[index] = cached
            self._depth_cache_order.append(index)
            if len(self._depth_cache_order) > self._depth_cache_limit:
                oldest = self._depth_cache_order.pop(0)
                self._depth_cache.pop(oldest, None)
        source_h, source_w = self.source_hw
        depth = torch.from_numpy(cached).float()
        if depth.shape != (source_h, source_w):
            depth = torch.nn.functional.interpolate(
                depth[None, None],
                size=(source_h, source_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        return resize_crop_tensor(
            depth,
            (source_h, source_w),
            target_hw,
            mode="bilinear",
        )

    def _candidate_target_starts(self, config: LongTrajectorySampleConfig) -> list[int]:
        rgb = set(self.rgb_by_index)
        calibration = set(self.pose_by_index) & set(self.intrinsics_by_index)
        depth = set(self.depth_location)
        available_capture = sorted(rgb & calibration & depth)
        if not available_capture:
            return []
        target_rgb_offsets = range(config.target_rgb_frames)
        target_latent_offsets = range(
            0,
            config.target_rgb_frames,
            config.vae_temporal_stride,
        )
        candidates: list[int] = []
        for start in available_capture:
            if start - available_capture[0] < config.history_min_frames:
                continue
            if not all(start + offset in rgb for offset in target_rgb_offsets):
                continue
            if not all(
                start + offset in calibration and start + offset in depth
                for offset in target_latent_offsets
            ):
                continue
            candidates.append(start)
        return candidates

    @staticmethod
    def _stratified_capture_indices(
        candidates: list[int],
        count: int,
    ) -> list[int]:
        if len(candidates) < count:
            raise ValueError(f"need {count} capture frames but only {len(candidates)} are valid")
        positions = np.linspace(0, len(candidates) - 1, count).round().astype(np.int64)
        selected = [candidates[int(position)] for position in positions]
        # linspace can repeat when numerical rounding meets short ranges.
        selected = sorted(set(selected))
        if len(selected) < count:
            for index in candidates:
                if index not in selected:
                    selected.append(index)
                    if len(selected) == count:
                        break
            selected.sort()
        return selected

    def sample(
        self,
        config: LongTrajectorySampleConfig,
        rng: random.Random,
    ) -> dict[str, torch.Tensor | str]:
        starts = self._candidate_target_starts(config)
        if not starts:
            raise RuntimeError(f"{self.root.name} has no valid long-trajectory training window")
        available_capture = sorted(
            set(self.rgb_by_index)
            & set(self.pose_by_index)
            & set(self.intrinsics_by_index)
            & set(self.depth_location)
        )
        target_start = rng.choice(starts)
        maximum_history = min(
            config.history_max_frames,
            target_start - available_capture[0],
        )
        minimum_history = min(config.history_min_frames, maximum_history)
        history_span = rng.randint(minimum_history, maximum_history)
        history_start = target_start - history_span
        history_candidates = [
            index
            for index in available_capture
            if history_start <= index <= target_start
        ]
        if target_start not in history_candidates:
            history_candidates.append(target_start)
            history_candidates.sort()
        capture_indices = self._stratified_capture_indices(
            history_candidates,
            config.capture_frames,
        )
        # The final capture is the clean one-latent overlap of the first chunk.
        if capture_indices[-1] != target_start:
            capture_indices[-1] = target_start
            capture_indices = sorted(set(capture_indices))
            if len(capture_indices) != config.capture_frames:
                capture_indices = self._stratified_capture_indices(
                    history_candidates[:-1],
                    config.capture_frames - 1,
                ) + [target_start]

        target_rgb_indices = list(
            range(target_start, target_start + config.target_rgb_frames)
        )
        target_latent_indices = target_rgb_indices[:: config.vae_temporal_stride]
        target_hw = (config.height, config.width)
        source_hw = self.source_hw

        capture_rgb = torch.stack(
            [self.read_rgb(index, target_hw) for index in capture_indices]
        )
        capture_depth = torch.stack(
            [self.read_depth(index, target_hw) for index in capture_indices]
        )
        target_rgb = torch.stack(
            [self.read_rgb(index, target_hw) for index in target_rgb_indices],
            dim=1,
        )
        target_depth = torch.stack(
            [self.read_depth(index, target_hw) for index in target_latent_indices]
        )

        capture_c2w = torch.from_numpy(
            np.stack([self.pose_by_index[index] for index in capture_indices])
        ).float()
        target_c2w = torch.from_numpy(
            np.stack([self.pose_by_index[index] for index in target_latent_indices])
        ).float()
        first_capture = capture_c2w[0]
        capture_c2w = normalize_c2w_to_first_capture(capture_c2w, first_capture)
        target_c2w = normalize_c2w_to_first_capture(target_c2w, first_capture)

        capture_intrinsics_vector = torch.from_numpy(
            np.stack([self.intrinsics_by_index[index] for index in capture_indices])
        ).float()
        target_intrinsics_vector = torch.from_numpy(
            np.stack([self.intrinsics_by_index[index] for index in target_latent_indices])
        ).float()
        capture_intrinsics = intrinsics_vector_to_matrix(
            resize_crop_intrinsics(capture_intrinsics_vector, source_hw, target_hw)
        )
        target_intrinsics = intrinsics_vector_to_matrix(
            resize_crop_intrinsics(target_intrinsics_vector, source_hw, target_hw)
        )

        capture_valid = (
            torch.isfinite(capture_depth)
            & (capture_depth >= config.min_depth)
            & (capture_depth <= config.max_depth)
        )
        target_valid = (
            torch.isfinite(target_depth)
            & (target_depth >= config.min_depth)
            & (target_depth <= config.max_depth)
        )
        return {
            "item_name": self.root.name,
            "capture_rgb": capture_rgb,
            "capture_depth": capture_depth,
            "capture_valid": capture_valid,
            "capture_c2w": capture_c2w,
            "capture_intrinsics": capture_intrinsics,
            "capture_indices": torch.tensor(capture_indices, dtype=torch.long),
            "target_rgb": target_rgb,
            "target_depth": target_depth,
            "target_valid": target_valid,
            "target_c2w": target_c2w,
            "target_intrinsics": target_intrinsics,
            "target_rgb_indices": torch.tensor(target_rgb_indices, dtype=torch.long),
            "target_latent_indices": torch.tensor(target_latent_indices, dtype=torch.long),
            "image_hw": torch.tensor(target_hw, dtype=torch.long),
        }


class RemoteVipeRoomTourDataset(IterableDataset):
    """Infinite item-reuse stream for long room-tour training."""

    def __init__(
        self,
        dataset_root: str,
        cache_root: str | Path,
        sample_config: LongTrajectorySampleConfig,
        *,
        rclone: Optional[RcloneConfig] = None,
        item_list: Optional[list[str]] = None,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        self.item_cache = RoomTourItemCache(dataset_root, cache_root, rclone=rclone)
        self.sample_config = sample_config
        self.items = item_list or self.item_cache.list_items()
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        shard_id = self.rank * workers + worker_id
        shard_count = self.world_size * workers
        items = self.items[shard_id::shard_count]
        if not items:
            raise RuntimeError(
                f"dataset shard {shard_id}/{shard_count} has no items; "
                "reduce DataLoader workers or ranks"
            )
        rng = random.Random(self.seed + 10_007 * shard_id)
        while True:
            shuffled = list(items)
            rng.shuffle(shuffled)
            for item_name in shuffled:
                try:
                    local_root = self.item_cache.materialize(item_name)
                    item = VipeRoomTourItem(local_root)
                except Exception:
                    logger.exception("failed to prepare room-tour item %s", item_name)
                    continue
                yielded = 0
                attempts = 0
                maximum_attempts = (
                    self.sample_config.samples_per_item
                    * self.sample_config.max_sample_attempts
                )
                while (
                    yielded < self.sample_config.samples_per_item
                    and attempts < maximum_attempts
                ):
                    attempts += 1
                    try:
                        yield item.sample(self.sample_config, rng)
                        yielded += 1
                    except Exception:
                        logger.exception(
                            "failed to sample %s (attempt %d)",
                            item_name,
                            attempts,
                        )
                        break
