from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .geometry import (
    center_resize_crop,
    intrinsics_vector_to_matrix,
    normalize_c2w_to_first_capture,
    resize_crop_intrinsics,
)

# ViPE artifacts and the usable RGB stream are indexed 0,5,10,... in the
# source video.  The whole GIM pipeline operates on a contiguous 0..N-1
# internal timeline after this conversion.
SOURCE_FRAME_STRIDE = 5


class LocalRoomTourIndex:
    """Resolve mounted room-tour scenes without copying their source files."""

    def __init__(self, dataset_root: str | Path) -> None:
        self.root = Path(dataset_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"local dataset_root is not a directory: {self.root}"
            )

    @staticmethod
    def _validate_item_name(item_name: str) -> str:
        item_name = item_name.strip().rstrip("/")
        if (
            not item_name
            or item_name in {".", ".."}
            or Path(item_name).name != item_name
        ):
            raise ValueError(
                f"item_name must be one direct child directory name, got {item_name!r}"
            )
        return item_name

    def list_items(self) -> list[str]:
        items = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not items:
            raise FileNotFoundError(
                f"no room-tour item directories below {self.root}"
            )
        return items

    def item_path(self, item_name: str) -> Path:
        item_name = self._validate_item_name(item_name)
        path = self.root / item_name
        if not path.is_dir():
            raise FileNotFoundError(
                f"room-tour item does not exist below dataset_root: {path}"
            )
        return path


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
    query_blocks: int = 2
    vae_temporal_stride: int = 4
    capture_min_rgb_frames: int = 257
    capture_max_rgb_frames: int = 801
    capture_curriculum_start_max_rgb_frames: int = 321
    capture_curriculum_epochs: int = 5
    capture_min_fraction_of_current_max: float = 0.75
    capture_query_guard_rgb_frames: int = 32
    trajectory_candidate_trials: int = 128
    trajectory_topk: int = 8
    trajectory_pose_stride: int = 4
    trajectory_rotation_weight: float = 0.25

    def validate(self) -> None:
        if self.height % 16 or self.width % 16:
            raise ValueError("height and width must be multiples of 16")
        if self.vae_temporal_stride < 1:
            raise ValueError("vae_temporal_stride must be positive")
        if self.target_rgb_frames < 2:
            raise ValueError("target_rgb_frames must be at least 2")
        if self.query_blocks < 1:
            raise ValueError("query_blocks must be positive")
        if (self.target_rgb_frames - 1) % self.vae_temporal_stride:
            raise ValueError(
                "target_rgb_frames must equal 1 + k * vae_temporal_stride"
            )
        if self.capture_min_rgb_frames < 2:
            raise ValueError("capture_min_rgb_frames must be at least 2")
        if self.capture_min_rgb_frames > self.capture_max_rgb_frames:
            raise ValueError(
                "capture_min_rgb_frames cannot exceed capture_max_rgb_frames"
            )
        if not (
            self.capture_min_rgb_frames
            <= self.capture_curriculum_start_max_rgb_frames
            <= self.capture_max_rgb_frames
        ):
            raise ValueError(
                "capture curriculum start max must lie inside capture bounds"
            )
        for name, value in (
            ("capture_min_rgb_frames", self.capture_min_rgb_frames),
            ("capture_max_rgb_frames", self.capture_max_rgb_frames),
            (
                "capture_curriculum_start_max_rgb_frames",
                self.capture_curriculum_start_max_rgb_frames,
            ),
        ):
            if (value - 1) % self.vae_temporal_stride:
                raise ValueError(
                    f"{name} must equal 1 + k * vae_temporal_stride"
                )
        if self.capture_curriculum_epochs < 0:
            raise ValueError("capture_curriculum_epochs cannot be negative")
        if not 0 < self.capture_min_fraction_of_current_max <= 1:
            raise ValueError(
                "capture_min_fraction_of_current_max must be in (0,1]"
            )
        if self.capture_query_guard_rgb_frames < 0:
            raise ValueError("capture_query_guard_rgb_frames cannot be negative")
        if self.trajectory_candidate_trials < 1 or self.trajectory_topk < 1:
            raise ValueError("trajectory candidate trials/topk must be positive")
        if self.trajectory_pose_stride < 1:
            raise ValueError("trajectory_pose_stride must be positive")
        if self.trajectory_rotation_weight < 0:
            raise ValueError("trajectory_rotation_weight cannot be negative")

    @property
    def target_latent_frames(self) -> int:
        return 1 + (self.target_rgb_frames - 1) // self.vae_temporal_stride

    @property
    def query_rgb_frames(self) -> int:
        return self.query_blocks * self.target_rgb_frames

    def capture_max_for_epoch(self, epoch: int) -> int:
        if self.capture_curriculum_epochs <= 1:
            return self.capture_max_rgb_frames
        progress = min(
            max(epoch, 0) / (self.capture_curriculum_epochs - 1),
            1.0,
        )
        raw = self.capture_curriculum_start_max_rgb_frames + progress * (
            self.capture_max_rgb_frames
            - self.capture_curriculum_start_max_rgb_frames
        )
        stride = self.vae_temporal_stride
        aligned = 1 + ((int(raw) - 1) // stride) * stride
        return max(self.capture_min_rgb_frames, aligned)

    def capture_bounds_for_epoch(self, epoch: int) -> tuple[int, int]:
        maximum = self.capture_max_for_epoch(epoch)
        stride = self.vae_temporal_stride
        fractional_minimum = int(
            maximum * self.capture_min_fraction_of_current_max
        )
        fractional_minimum = (
            1
            + (
                (fractional_minimum - 1 + stride - 1)
                // stride
            )
            * stride
        )
        minimum = max(self.capture_min_rgb_frames, fractional_minimum)
        return min(minimum, maximum), maximum


@dataclass(frozen=True)
class RoomTourSample:
    item_name: str
    local_root: str
    epoch: int
    capture_rgb_indices: tuple[int, ...]
    query_rgb_blocks: tuple[tuple[int, ...], ...]
    geometry_query_indices: tuple[int, ...]
    trajectory_overlap_score: float
    image_hw: tuple[int, int]

    @property
    def capture_start(self) -> int:
        return self.capture_rgb_indices[0]

    @property
    def query_start(self) -> int:
        return self.query_rgb_blocks[0][0]


class VipeRoomTourItem:
    """Indexed view of one sparse ViPE room tour on the local filesystem."""

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

    def _trajectory_overlap_score(
        self,
        capture: tuple[int, ...],
        query: tuple[int, ...],
        config: GeometryMemorySampleConfig,
    ) -> float:
        stride = config.trajectory_pose_stride
        capture_indices = capture[::stride]
        query_indices = query[::stride]
        capture_pose = np.stack(
            [self.pose_by_index[index] for index in capture_indices]
        )
        query_pose = np.stack(
            [self.pose_by_index[index] for index in query_indices]
        )
        all_positions = np.stack(
            [self.pose_by_index[index][:3, 3] for index in self.indices]
        )
        low, high = np.quantile(all_positions, (0.05, 0.95), axis=0)
        position_scale = max(float(np.linalg.norm(high - low)), 1e-4)

        capture_position = capture_pose[:, :3, 3]
        query_position = query_pose[:, :3, 3]
        position_distance = np.linalg.norm(
            query_position[:, None] - capture_position[None],
            axis=-1,
        ) / position_scale

        capture_forward = capture_pose[:, :3, 2]
        query_forward = query_pose[:, :3, 2]
        capture_forward /= np.clip(
            np.linalg.norm(capture_forward, axis=-1, keepdims=True),
            1e-8,
            None,
        )
        query_forward /= np.clip(
            np.linalg.norm(query_forward, axis=-1, keepdims=True),
            1e-8,
            None,
        )
        cosine = np.clip(query_forward @ capture_forward.T, -1.0, 1.0)
        angle = np.arccos(cosine) / np.pi
        pair_cost = (
            position_distance
            + config.trajectory_rotation_weight * angle
        )
        nearest = pair_cost.min(axis=1)
        return float(np.median(nearest) + 0.25 * np.quantile(nearest, 0.9))

    def _random_pair_candidate(
        self,
        config: GeometryMemorySampleConfig,
        rng: random.Random,
        epoch: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        first, last = self.indices[0], self.indices[-1]
        query_length = config.query_rgb_frames
        guard = config.capture_query_guard_rgb_frames
        configured_minimum, configured_maximum = (
            config.capture_bounds_for_epoch(epoch)
        )
        maximum = min(
            configured_maximum,
            len(self.indices) - query_length - guard,
        )
        stride = config.vae_temporal_stride
        maximum = 1 + ((maximum - 1) // stride) * stride
        lengths = list(
            range(configured_minimum, maximum + 1, stride)
        )
        if not lengths:
            raise RuntimeError(
                f"{self.root.name} is too short for capture>="
                f"{configured_minimum}, query={query_length}, "
                f"guard={guard}"
            )
        capture_length = rng.choice(lengths)
        # Choose the side on which the independent query trajectory lies
        # before drawing a capture start. This guarantees that every sampled
        # capture leaves enough room for the complete query and guard.
        latest_capture_start = last - capture_length + 1
        sides: list[tuple[str, int, int]] = []
        before_minimum = first + query_length + guard
        if before_minimum <= latest_capture_start:
            sides.append(("before", before_minimum, latest_capture_start))
        after_maximum = last - capture_length - guard - query_length + 1
        if first <= after_maximum:
            sides.append(("after", first, after_maximum))
        if not sides:
            raise RuntimeError("no isolated capture/query placement exists")
        side, capture_start_min, capture_start_max = rng.choice(sides)
        capture_start = rng.randint(capture_start_min, capture_start_max)
        capture_end = capture_start + capture_length - 1

        if side == "before":
            query_start = rng.randint(
                first,
                capture_start - guard - query_length,
            )
        else:
            query_start = rng.randint(
                capture_end + guard + 1,
                last - query_length + 1,
            )
        capture = tuple(
            range(capture_start, capture_start + capture_length)
        )
        query = tuple(range(query_start, query_start + query_length))
        return capture, query

    def make_sample(
        self,
        config: GeometryMemorySampleConfig,
        rng: random.Random,
        *,
        epoch: int = 0,
        capture_start: Optional[int] = None,
        query_start: Optional[int] = None,
        capture_rgb_frames: Optional[int] = None,
    ) -> RoomTourSample:
        config.validate()
        if (capture_start is None) != (query_start is None):
            raise ValueError(
                "capture_start and query_start must be supplied together"
            )
        if capture_start is not None:
            capture_length = (
                config.capture_max_for_epoch(epoch)
                if capture_rgb_frames is None
                else int(capture_rgb_frames)
            )
            if (capture_length - 1) % config.vae_temporal_stride:
                raise ValueError(
                    "capture_rgb_frames must equal 1 + k * vae_temporal_stride"
                )
            capture = tuple(
                range(int(capture_start), int(capture_start) + capture_length)
            )
            query = tuple(
                range(int(query_start), int(query_start) + config.query_rgb_frames)
            )
            missing = (set(capture) | set(query)) - set(self.indices)
            if missing:
                raise ValueError(
                    f"explicit trajectory indices are out of range: "
                    f"{sorted(missing)[:8]}"
                )
            gap = max(
                query[0] - capture[-1] - 1,
                capture[0] - query[-1] - 1,
            )
            if gap < config.capture_query_guard_rgb_frames:
                raise ValueError(
                    "capture/query windows overlap or violate the temporal guard"
                )
            candidates = [
                (
                    self._trajectory_overlap_score(
                        capture,
                        query,
                        config,
                    ),
                    capture,
                    query,
                )
            ]
        else:
            unique: dict[
                tuple[int, int, int],
                tuple[float, tuple[int, ...], tuple[int, ...]],
            ] = {}
            for _ in range(config.trajectory_candidate_trials):
                capture, query = self._random_pair_candidate(
                    config,
                    rng,
                    epoch,
                )
                key = (capture[0], len(capture), query[0])
                if key not in unique:
                    unique[key] = (
                        self._trajectory_overlap_score(
                            capture,
                            query,
                            config,
                        ),
                        capture,
                        query,
                    )
            candidates = sorted(unique.values(), key=lambda value: value[0])
        top = candidates[: min(config.trajectory_topk, len(candidates))]
        overlap_score, capture, query = rng.choice(top)
        blocks = tuple(
            tuple(
                query[
                    block * config.target_rgb_frames :
                    (block + 1) * config.target_rgb_frames
                ]
            )
            for block in range(config.query_blocks)
        )
        geometry_queries = tuple(rng.choice(block) for block in blocks)
        return RoomTourSample(
            item_name=self.root.name,
            local_root=str(self.root),
            epoch=int(epoch),
            capture_rgb_indices=capture,
            query_rgb_blocks=blocks,
            geometry_query_indices=geometry_queries,
            trajectory_overlap_score=overlap_score,
            image_hw=(config.height, config.width),
        )


class LocalVipeRoomTourDataset(Dataset[RoomTourSample]):
    """One randomized capture/query trajectory pair per scene and epoch."""

    def __init__(
        self,
        dataset_root: str | Path,
        sample_config: GeometryMemorySampleConfig,
        *,
        item_list: Optional[list[str]] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.item_index = LocalRoomTourIndex(dataset_root)
        self.sample_config = sample_config
        self.items = (
            self.item_index.list_items()
            if item_list is None
            else list(item_list)
        )
        if not self.items:
            raise ValueError("item_list cannot be empty")
        for item_name in self.items:
            self.item_index.item_path(item_name)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> RoomTourSample:
        item_name = self.items[int(index)]
        item = VipeRoomTourItem(self.item_index.item_path(item_name))
        rng = random.Random(
            self.seed + 1_000_003 * self.epoch + 10_007 * int(index)
        )
        return item.make_sample(
            self.sample_config,
            rng,
            epoch=self.epoch,
        )
