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

# VIPE estimates depth/pose/intrinsics once every five source RGB frames.  All
# four modalities store source indices 0,5,10,...; normalize them to the
# contiguous internal training timeline 0,1,2,... so the rest of the sampler
# does not need sparse-index special cases.
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
    capture_clips: int = 16
    capture_clip_rgb_frames: int = 9
    preceding_rgb_frames: int = 8
    reference_frames: int = 4
    history_min_frames: int = 256
    history_max_frames: int = 4096
    latent_frames_per_chunk: int = 9
    vae_temporal_stride: int = 4
    samples_per_item: int = 32
    min_depth: float = 0.1
    max_depth: float = 20.0
    max_sample_attempts: int = 64

    @property
    def capture_latent_frames_per_clip(self) -> int:
        return 1 + (
            self.capture_clip_rgb_frames - 1
        ) // self.vae_temporal_stride

    @property
    def target_rgb_frames(self) -> int:
        return 1 + (
            self.latent_frames_per_chunk - 1
        ) * self.vae_temporal_stride

    def validate(self) -> None:
        if self.capture_clips < 1:
            raise ValueError("capture_clips must be positive")
        if self.reference_frames < 0:
            raise ValueError("reference_frames must be non-negative")
        if self.preceding_rgb_frames < 1:
            raise ValueError("preceding_rgb_frames must be positive")
        if (self.capture_clip_rgb_frames - 1) % self.vae_temporal_stride:
            raise ValueError(
                "capture_clip_rgb_frames must equal 1 + k * vae_temporal_stride"
            )
        if self.preceding_rgb_frames % self.vae_temporal_stride:
            raise ValueError(
                "preceding_rgb_frames must be divisible by vae_temporal_stride"
            )
        if self.capture_clip_rgb_frames != self.preceding_rgb_frames + 1:
            raise ValueError(
                "the final capture clip must contain preceding_rgb_frames + "
                "the clean target overlap frame"
            )


def _load_npz_mapping(
    path: Path,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    output: dict[int, np.ndarray] = {}
    source_indices: dict[int, int] = {}
    with np.load(path) as payload:
        for index, value in zip(payload["inds"], payload["data"], strict=True):
            source_index = int(index)
            if source_index % SOURCE_FRAME_STRIDE:
                continue
            internal_index = source_index // SOURCE_FRAME_STRIDE
            output[internal_index] = np.asarray(value)
            source_indices[internal_index] = source_index
    if not output:
        raise ValueError(
            f"{path} has no indices divisible by {SOURCE_FRAME_STRIDE}"
        )
    return output, source_indices


def _decode_exr(payload: bytes) -> np.ndarray:
    """Decode the metric Z channel written by VIPE with the OpenEXR bindings."""

    with tempfile.NamedTemporaryFile(suffix=".exr") as temporary:
        temporary.write(payload)
        temporary.flush()
        try:
            import OpenEXR
        except ImportError as exc:
            raise RuntimeError(
                "Depth EXR decoding requires the OpenEXR Python package. Install "
                "the training dependencies with `python -m pip install -r "
                "requirements-training.txt`, or install it directly with "
                "`python -m pip install 'OpenEXR>=3.2'`."
            ) from exc

        # OpenEXR >=3.3 exposes channels directly as NumPy arrays and no longer
        # requires callers to construct an Imath pixel type.
        if hasattr(OpenEXR, "File"):
            with OpenEXR.File(temporary.name) as file:
                channels = file.channels()
                if "Z" not in channels:
                    raise ValueError(
                        f"VIPE depth EXR has no Z channel; found {sorted(channels)}"
                    )
                depth = np.asarray(channels["Z"].pixels)
        else:
            # OpenEXR 3.2 and older bindings use the legacy InputFile API.
            try:
                import Imath
            except ImportError as exc:
                raise RuntimeError(
                    "The installed legacy OpenEXR bindings require Imath. "
                    "Upgrade with `python -m pip install --upgrade 'OpenEXR>=3.3'`."
                ) from exc
            file = OpenEXR.InputFile(temporary.name)
            try:
                window = file.header()["dataWindow"]
                width = int(window.max.x - window.min.x + 1)
                height = int(window.max.y - window.min.y + 1)
                raw = file.channel("Z", Imath.PixelType(Imath.PixelType.FLOAT))
                depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
            finally:
                file.close()

    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"VIPE Z channel must be 2D, got shape {depth.shape}")
    return depth.astype(np.float32, copy=True)


class VipeRoomTourItem:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.metadata = self._load_metadata()
        artifact_root = self.root / "vipe" / "vipe_artifacts"
        (
            self.pose_by_index,
            self.pose_source_index_by_index,
        ) = _load_npz_mapping(artifact_root / "pose" / "video.npz")
        (
            self.intrinsics_by_index,
            self.intrinsics_source_index_by_index,
        ) = _load_npz_mapping(
            artifact_root / "intrinsics" / "video.npz"
        )
        self._validate_camera_type(artifact_root / "intrinsics" / "video_camera.txt")
        (
            self.rgb_by_index,
            self.rgb_source_index_by_index,
        ) = self._index_rgb()
        (
            self.depth_location,
            self.depth_source_index_by_index,
        ) = self._index_depth(artifact_root / "depth")
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

    def _index_rgb(self) -> tuple[dict[int, Path], dict[int, int]]:
        """Map contiguous VIPE indices to every fifth source RGB frame."""

        output: dict[int, Path] = {}
        source_indices: dict[int, int] = {}
        for path in sorted((self.root / "RGB").glob("*")):
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            try:
                source_index = int(path.stem)
            except ValueError:
                continue
            if source_index % SOURCE_FRAME_STRIDE:
                continue
            internal_index = source_index // SOURCE_FRAME_STRIDE
            output[internal_index] = path
            source_indices[internal_index] = source_index
        if not output:
            raise FileNotFoundError(
                f"no RGB frames divisible by {SOURCE_FRAME_STRIDE} below "
                f"{self.root / 'RGB'}"
            )
        return output, source_indices

    @staticmethod
    def _index_depth(
        depth_root: Path,
    ) -> tuple[dict[int, tuple[Path, str]], dict[int, int]]:
        final_archive = depth_root / "video.zip"
        archives: list[Path] = []
        if final_archive.is_file():
            archives.append(final_archive)
        parts = depth_root / "video.zip.parts"
        if parts.is_dir():
            archives.extend(sorted(parts.glob("*.zip")))
        output: dict[int, tuple[Path, str]] = {}
        source_indices: dict[int, int] = {}
        for archive_path in archives:
            try:
                with zipfile.ZipFile(archive_path, "r") as archive:
                    for member in archive.namelist():
                        try:
                            source_index = int(Path(member).stem)
                        except ValueError:
                            continue
                        if source_index % SOURCE_FRAME_STRIDE:
                            continue
                        internal_index = source_index // SOURCE_FRAME_STRIDE
                        output.setdefault(internal_index, (archive_path, member))
                        source_indices.setdefault(internal_index, source_index)
            except zipfile.BadZipFile:
                logger.warning("ignoring incomplete depth archive %s", archive_path)
        if not output:
            raise FileNotFoundError(f"no complete depth frames below {depth_root}")
        return output, source_indices

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
        config.validate()
        rgb = set(self.rgb_by_index)
        calibration = set(self.pose_by_index) & set(self.intrinsics_by_index)
        depth = set(self.depth_location)
        geometry = calibration & depth
        available = sorted(rgb & geometry)
        if not available:
            return []
        target_rgb_offsets = tuple(range(config.target_rgb_frames))
        target_latent_offsets = tuple(
            range(0, config.target_rgb_frames, config.vae_temporal_stride)
        )
        preceding_rgb_offsets = tuple(range(-config.preceding_rgb_frames, 0))
        preceding_latent_offsets = tuple(
            range(
                -config.preceding_rgb_frames,
                0,
                config.vae_temporal_stride,
            )
        )
        final_capture_start_offset = 1 - config.capture_clip_rgb_frames
        capture_anchor_offsets = tuple(
            range(
                final_capture_start_offset,
                1,
                config.vae_temporal_stride,
            )
        )
        candidates: list[int] = []
        for start in available:
            if start - available[0] < config.history_min_frames:
                continue
            if not all(start + offset in rgb for offset in target_rgb_offsets):
                continue
            if not all(start + offset in rgb for offset in preceding_rgb_offsets):
                continue
            if not all(start + offset in geometry for offset in target_latent_offsets):
                continue
            if not all(start + offset in geometry for offset in preceding_latent_offsets):
                continue
            if not all(
                start + offset in rgb
                for offset in range(
                    final_capture_start_offset,
                    1,
                )
            ):
                continue
            if not all(start + offset in geometry for offset in capture_anchor_offsets):
                continue
            candidates.append(start)
        return candidates

    def _valid_capture_clip_starts(
        self,
        *,
        history_start: int,
        target_start: int,
        config: LongTrajectorySampleConfig,
    ) -> list[int]:
        rgb = set(self.rgb_by_index)
        geometry = (
            set(self.pose_by_index)
            & set(self.intrinsics_by_index)
            & set(self.depth_location)
        )
        anchor_offsets = tuple(
            range(
                0,
                config.capture_clip_rgb_frames,
                config.vae_temporal_stride,
            )
        )
        latest_start = target_start - config.capture_clip_rgb_frames + 1
        starts = []
        for start in range(history_start, latest_start + 1):
            if not all(
                start + offset in rgb
                for offset in range(config.capture_clip_rgb_frames)
            ):
                continue
            if not all(start + offset in geometry for offset in anchor_offsets):
                continue
            starts.append(start)
        return starts

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
        available = sorted(
            set(self.rgb_by_index)
            & set(self.pose_by_index)
            & set(self.intrinsics_by_index)
            & set(self.depth_location)
        )
        target_start = rng.choice(starts)
        maximum_history = min(
            config.history_max_frames,
            target_start - available[0],
        )
        minimum_history = min(config.history_min_frames, maximum_history)
        history_span = rng.randint(minimum_history, maximum_history)
        history_start = target_start - history_span
        capture_start_candidates = self._valid_capture_clip_starts(
            history_start=history_start,
            target_start=target_start,
            config=config,
        )
        final_capture_start = target_start - config.capture_clip_rgb_frames + 1
        if final_capture_start not in capture_start_candidates:
            raise RuntimeError("the target overlap capture clip is incomplete")
        if len(capture_start_candidates) < config.capture_clips:
            raise RuntimeError(
                f"only {len(capture_start_candidates)} valid capture clips for "
                f"requested {config.capture_clips}"
            )
        capture_clip_starts = self._stratified_capture_indices(
            capture_start_candidates,
            config.capture_clips,
        )
        if capture_clip_starts[-1] != final_capture_start:
            capture_clip_starts = self._stratified_capture_indices(
                capture_start_candidates[:-1],
                config.capture_clips - 1,
            ) + [final_capture_start]

        capture_anchor_offsets = tuple(
            range(
                0,
                config.capture_clip_rgb_frames,
                config.vae_temporal_stride,
            )
        )
        capture_indices = [
            start + offset
            for start in capture_clip_starts
            for offset in capture_anchor_offsets
        ]

        target_rgb_indices = list(
            range(target_start, target_start + config.target_rgb_frames)
        )
        target_latent_indices = target_rgb_indices[:: config.vae_temporal_stride]
        preceding_rgb_indices = list(
            range(target_start - config.preceding_rgb_frames, target_start)
        )
        preceding_latent_indices = preceding_rgb_indices[
            :: config.vae_temporal_stride
        ]
        target_hw = (config.height, config.width)
        source_hw = self.source_hw

        capture_rgb = torch.stack(
            [
                torch.stack(
                    [
                        self.read_rgb(start + offset, target_hw)
                        for offset in range(config.capture_clip_rgb_frames)
                    ],
                    dim=1,
                )
                for start in capture_clip_starts
            ]
        )
        capture_depth = torch.stack(
            [self.read_depth(index, target_hw) for index in capture_indices]
        )
        preceding_rgb = torch.stack(
            [self.read_rgb(index, target_hw) for index in preceding_rgb_indices],
            dim=1,
        )
        preceding_depth = torch.stack(
            [self.read_depth(index, target_hw) for index in preceding_latent_indices]
        )
        target_rgb = torch.stack(
            [self.read_rgb(index, target_hw) for index in target_rgb_indices],
            dim=1,
        )
        target_depth = torch.stack(
            [self.read_depth(index, target_hw) for index in target_latent_indices]
        )

        capture_c2w_raw = torch.from_numpy(
            np.stack([self.pose_by_index[index] for index in capture_indices])
        ).float()
        preceding_c2w_raw = torch.from_numpy(
            np.stack([self.pose_by_index[index] for index in preceding_latent_indices])
        ).float()
        target_c2w = torch.from_numpy(
            np.stack([self.pose_by_index[index] for index in target_latent_indices])
        ).float()
        first_capture = capture_c2w_raw[0]
        capture_c2w = normalize_c2w_to_first_capture(
            capture_c2w_raw,
            first_capture,
        )
        preceding_c2w = normalize_c2w_to_first_capture(
            preceding_c2w_raw,
            first_capture,
        )
        target_c2w = normalize_c2w_to_first_capture(target_c2w, first_capture)

        capture_intrinsics_vector = torch.from_numpy(
            np.stack([self.intrinsics_by_index[index] for index in capture_indices])
        ).float()
        preceding_intrinsics_vector = torch.from_numpy(
            np.stack(
                [
                    self.intrinsics_by_index[index]
                    for index in preceding_latent_indices
                ]
            )
        ).float()
        target_intrinsics_vector = torch.from_numpy(
            np.stack([self.intrinsics_by_index[index] for index in target_latent_indices])
        ).float()
        capture_intrinsics = intrinsics_vector_to_matrix(
            resize_crop_intrinsics(capture_intrinsics_vector, source_hw, target_hw)
        )
        preceding_intrinsics = intrinsics_vector_to_matrix(
            resize_crop_intrinsics(
                preceding_intrinsics_vector,
                source_hw,
                target_hw,
            )
        )
        target_intrinsics = intrinsics_vector_to_matrix(
            resize_crop_intrinsics(target_intrinsics_vector, source_hw, target_hw)
        )

        capture_valid = (
            torch.isfinite(capture_depth)
            & (capture_depth >= config.min_depth)
            & (capture_depth <= config.max_depth)
        )
        preceding_valid = (
            torch.isfinite(preceding_depth)
            & (preceding_depth >= config.min_depth)
            & (preceding_depth <= config.max_depth)
        )
        target_valid = (
            torch.isfinite(target_depth)
            & (target_depth >= config.min_depth)
            & (target_depth <= config.max_depth)
        )

        reference_candidates = [
            index
            for index in available
            if history_start <= index < target_start
        ]
        reference_indices = (
            self._stratified_capture_indices(
                reference_candidates,
                config.reference_frames,
            )
            if config.reference_frames
            else []
        )
        if reference_indices:
            reference_rgb = torch.stack(
                [self.read_rgb(index, target_hw) for index in reference_indices]
            )
        else:
            reference_rgb = torch.empty(
                0,
                3,
                target_hw[0],
                target_hw[1],
            )
        return {
            "item_name": self.root.name,
            "capture_rgb": capture_rgb,
            "capture_depth": capture_depth,
            "capture_valid": capture_valid,
            "capture_c2w": capture_c2w,
            "capture_intrinsics": capture_intrinsics,
            "capture_indices": torch.tensor(capture_indices, dtype=torch.long),
            "capture_clip_starts": torch.tensor(
                capture_clip_starts,
                dtype=torch.long,
            ),
            "preceding_rgb": preceding_rgb,
            "preceding_depth": preceding_depth,
            "preceding_valid": preceding_valid,
            "preceding_c2w": preceding_c2w,
            "preceding_intrinsics": preceding_intrinsics,
            "preceding_rgb_indices": torch.tensor(
                preceding_rgb_indices,
                dtype=torch.long,
            ),
            "preceding_latent_indices": torch.tensor(
                preceding_latent_indices,
                dtype=torch.long,
            ),
            "reference_rgb": reference_rgb,
            "reference_indices": torch.tensor(
                reference_indices,
                dtype=torch.long,
            ),
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
