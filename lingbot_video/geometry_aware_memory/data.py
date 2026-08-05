from __future__ import annotations

import json
import random
import re
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
_ITEM_SOURCE_RANGE = re.compile(r"_(\d+)_(\d+)\.mp4$")


def estimate_sparse_frame_count(item_name: str) -> Optional[int]:
    """Estimate the usable 0,5,10,... RGB count from a scene directory name.

    Dataset names end in ``_<source_start>_<source_end>.mp4``.  Integer
    division is deliberately conservative when the source span is not
    divisible by five, so the fast prefilter never admits a borderline short
    scene based on a one-frame overestimate.
    """

    match = _ITEM_SOURCE_RANGE.search(Path(item_name).name)
    if match is None:
        return None
    source_start, source_end = (int(value) for value in match.groups())
    if source_end <= source_start:
        return 0
    return (source_end - source_start) // SOURCE_FRAME_STRIDE


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
    target_rgb_frames: int = 41
    query_blocks: int = 1
    local_window_rgb_frames: int = 81
    memory_views_min: int = 2
    memory_views_max: int = 24
    retrieval_rotation_weight: float = 0.25
    retrieval_temperature: float = 0.25

    def validate(self) -> None:
        if self.height % 16 or self.width % 16:
            raise ValueError("height and width must be multiples of 16")
        if self.target_rgb_frames < 2:
            raise ValueError("target_rgb_frames must be at least 2")
        if self.query_blocks < 1:
            raise ValueError("query_blocks must be positive")
        if self.local_window_rgb_frames < self.query_rgb_frames + 2:
            raise ValueError(
                "local_window_rgb_frames must leave at least two non-target "
                "memory candidates"
            )
        candidate_count = self.local_window_rgb_frames - self.query_rgb_frames
        if self.memory_views_min < 2:
            raise ValueError("memory_views_min must be at least 2")
        if self.memory_views_max < self.memory_views_min:
            raise ValueError(
                "memory_views_max cannot be smaller than memory_views_min"
            )
        if self.memory_views_max > candidate_count:
            raise ValueError(
                "memory_views_max cannot exceed the number of non-target "
                f"frames in the local window ({candidate_count})"
            )
        if self.retrieval_rotation_weight < 0:
            raise ValueError("retrieval_rotation_weight cannot be negative")
        if self.retrieval_temperature <= 0:
            raise ValueError("retrieval_temperature must be positive")

    @property
    def target_latent_frames(self) -> int:
        # Every RGB frame is encoded as an independent one-frame Wan sample.
        return self.target_rgb_frames

    @property
    def query_rgb_frames(self) -> int:
        return self.query_blocks * self.target_rgb_frames


def has_local_retrieval_sample_for_frame_count(
    frame_count: int,
    config: GeometryMemorySampleConfig,
) -> bool:
    """Whether the sparse internal timeline can supply one local window."""

    return int(frame_count) >= config.local_window_rgb_frames


@dataclass(frozen=True)
class RoomTourSample:
    item_name: str
    local_root: str
    epoch: int
    local_window_start: int
    local_window_end: int
    capture_rgb_indices: tuple[int, ...]
    query_rgb_blocks: tuple[tuple[int, ...], ...]
    geometry_query_indices: tuple[int, ...]
    retrieval_coverage_score: float
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

    def _candidate_target_similarity(
        self,
        candidates: tuple[int, ...],
        query: tuple[int, ...],
        config: GeometryMemorySampleConfig,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pose-only candidate relevance to every target view.

        Translation is normalized by the robust spatial extent of this local
        window. Rotation uses the geodesic SO(3) angle. The returned
        similarity matrix is consumed by greedy facility-location retrieval,
        so a redundant view has almost no marginal gain once another selected
        view already covers the same target cameras.
        """

        candidate_pose = np.stack(
            [self.pose_by_index[index] for index in candidates]
        ).astype(np.float64)
        query_pose = np.stack(
            [self.pose_by_index[index] for index in query]
        ).astype(np.float64)
        positions = np.concatenate(
            (candidate_pose[:, :3, 3], query_pose[:, :3, 3]),
            axis=0,
        )
        low, high = np.quantile(positions, (0.05, 0.95), axis=0)
        position_scale = max(float(np.linalg.norm(high - low)), 1e-4)
        position_cost = np.linalg.norm(
            candidate_pose[:, None, :3, 3]
            - query_pose[None, :, :3, 3],
            axis=-1,
        ) / position_scale

        candidate_rotation = candidate_pose[:, :3, :3]
        query_rotation = query_pose[:, :3, :3]
        relative = np.einsum(
            "cij,tjk->ctik",
            np.swapaxes(candidate_rotation, -1, -2),
            query_rotation,
        )
        trace = np.trace(relative, axis1=-2, axis2=-1)
        rotation_cost = np.arccos(
            np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
        ) / np.pi
        pair_cost = (
            position_cost
            + config.retrieval_rotation_weight * rotation_cost
        )
        similarity = np.exp(-pair_cost / config.retrieval_temperature)
        return similarity, pair_cost

    def _retrieve_memory_views(
        self,
        candidates: tuple[int, ...],
        query: tuple[int, ...],
        view_count: int,
        config: GeometryMemorySampleConfig,
        rng: random.Random,
    ) -> tuple[tuple[int, ...], float, float]:
        if not 1 <= view_count <= len(candidates):
            raise ValueError(
                f"memory view count {view_count} is invalid for "
                f"{len(candidates)} candidates"
            )
        similarity, pair_cost = self._candidate_target_similarity(
            candidates,
            query,
            config,
        )
        best_coverage = np.zeros(len(query), dtype=np.float64)
        remaining = list(range(len(candidates)))
        selected: list[int] = []
        # Seeded jitter only breaks exact ties; it does not change meaningful
        # facility-location gains.
        tie_break = np.asarray(
            [rng.random() for _ in candidates],
            dtype=np.float64,
        ) * 1e-12
        for _ in range(view_count):
            gains = np.asarray(
                [
                    np.maximum(best_coverage, similarity[index]).mean()
                    - best_coverage.mean()
                    + tie_break[index]
                    for index in remaining
                ]
            )
            chosen_position = int(np.argmax(gains))
            chosen = remaining.pop(chosen_position)
            selected.append(chosen)
            best_coverage = np.maximum(
                best_coverage,
                similarity[chosen],
            )
        selected_indices = tuple(
            sorted(candidates[index] for index in selected)
        )
        selected_cost = pair_cost[np.asarray(selected)].min(axis=0)
        overlap_score = float(
            np.median(selected_cost) + 0.25 * np.quantile(selected_cost, 0.9)
        )
        return (
            selected_indices,
            float(best_coverage.mean()),
            overlap_score,
        )

    def make_sample(
        self,
        config: GeometryMemorySampleConfig,
        rng: random.Random,
        *,
        epoch: int = 0,
        local_window_start: Optional[int] = None,
        target_start: Optional[int] = None,
        memory_view_count: Optional[int] = None,
    ) -> RoomTourSample:
        config.validate()
        explicit = (
            local_window_start,
            target_start,
            memory_view_count,
        )
        if any(value is not None for value in explicit) and not all(
            value is not None for value in explicit
        ):
            raise ValueError(
                "explicit sampling requires local_window_start, target_start, "
                "and memory_view_count together"
            )
        first = self.indices[0]
        last_window_start = (
            self.indices[-1] - config.local_window_rgb_frames + 1
        )
        if last_window_start < first:
            raise RuntimeError(
                f"{self.root.name} has {len(self.indices)} usable frames but "
                f"local_window_rgb_frames={config.local_window_rgb_frames}"
            )
        if local_window_start is None:
            window_start = rng.randint(first, last_window_start)
            memory_count = rng.randint(
                config.memory_views_min,
                config.memory_views_max,
            )
            context_frames = (
                config.local_window_rgb_frames - config.query_rgb_frames
            )
            # Centering supplies capture context on both sides of the withheld
            # target trajectory, avoiding the old pure-extrapolation setup.
            query_start = window_start + context_frames // 2
        else:
            window_start = int(local_window_start)
            query_start = int(target_start)
            memory_count = int(memory_view_count)
        window_end = window_start + config.local_window_rgb_frames - 1
        query = tuple(
            range(query_start, query_start + config.query_rgb_frames)
        )
        if window_start < first or window_end > self.indices[-1]:
            raise ValueError(
                f"local window {window_start}:{window_end} is out of range"
            )
        if query[0] < window_start or query[-1] > window_end:
            raise ValueError(
                "target trajectory must lie inside the local window"
            )
        query_set = set(query)
        candidates = tuple(
            index
            for index in range(window_start, window_end + 1)
            if index not in query_set
        )
        if not (
            config.memory_views_min
            <= memory_count
            <= config.memory_views_max
        ):
            raise ValueError(
                f"memory_view_count must be in "
                f"[{config.memory_views_min}, {config.memory_views_max}]"
            )
        capture, coverage_score, overlap_score = self._retrieve_memory_views(
            candidates,
            query,
            memory_count,
            config,
            rng,
        )
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
            local_window_start=window_start,
            local_window_end=window_end,
            capture_rgb_indices=capture,
            query_rgb_blocks=blocks,
            geometry_query_indices=geometry_queries,
            retrieval_coverage_score=coverage_score,
            trajectory_overlap_score=overlap_score,
            image_hw=(config.height, config.width),
        )


class LocalVipeRoomTourDataset(Dataset[RoomTourSample]):
    """One local target-and-retrieved-memory sample per scene and epoch."""

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
        items = (
            self.item_index.list_items()
            if item_list is None
            else list(item_list)
        )
        if not items:
            raise ValueError("item_list cannot be empty")
        for item_name in items:
            self.item_index.item_path(item_name)
        self.all_items = tuple(items)
        self._estimated_frame_counts = tuple(
            estimate_sparse_frame_count(item_name)
            for item_name in self.all_items
        )
        self.items: list[str] = []
        self._active_ordinals: list[int] = []
        self.skipped_items: tuple[str, ...] = ()
        self.seed = int(seed)
        self.epoch = -1
        self.set_epoch(0)

    @property
    def total_item_count(self) -> int:
        return len(self.all_items)

    @property
    def skipped_item_count(self) -> int:
        return len(self.skipped_items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        active: list[tuple[int, str]] = []
        skipped: list[str] = []
        for ordinal, (item_name, estimated_frames) in enumerate(
            zip(
                self.all_items,
                self._estimated_frame_counts,
                strict=True,
            )
        ):
            # Non-standard names cannot be prefiltered cheaply. Keep them and
            # let VipeRoomTourItem's exact common-modality index validate them.
            if (
                estimated_frames is None
                or has_local_retrieval_sample_for_frame_count(
                    estimated_frames,
                    self.sample_config,
                )
            ):
                active.append((ordinal, item_name))
            else:
                skipped.append(item_name)
        if not active:
            raise RuntimeError(
                "no dataset item can supply a local retrieval sample with "
                f"window_rgb={self.sample_config.local_window_rgb_frames}, "
                f"query_rgb={self.sample_config.query_rgb_frames}"
            )
        self._active_ordinals = [ordinal for ordinal, _ in active]
        self.items = [item_name for _, item_name in active]
        self.skipped_items = tuple(skipped)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> RoomTourSample:
        active_index = int(index)
        item_name = self.items[active_index]
        stable_ordinal = self._active_ordinals[active_index]
        item = VipeRoomTourItem(self.item_index.item_path(item_name))
        rng = random.Random(
            self.seed + 1_000_003 * self.epoch + 10_007 * stable_ordinal
        )
        return item.make_sample(
            self.sample_config,
            rng,
            epoch=self.epoch,
        )
