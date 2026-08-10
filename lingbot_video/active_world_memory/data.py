from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .geometry import (
    intrinsics_vector_to_matrix,
    normalize_c2w_to_first_capture,
    resize_crop_intrinsics,
)


SOURCE_FRAME_STRIDE = 5
_ITEM_SOURCE_RANGE = re.compile(r"_(\d+)_(\d+)\.mp4$")


def estimate_sparse_frame_count(item_name: str) -> Optional[int]:
    match = _ITEM_SOURCE_RANGE.search(Path(item_name).name)
    if match is None:
        return None
    start, end = (int(value) for value in match.groups())
    return max(0, (end - start) // SOURCE_FRAME_STRIDE)


def _load_npz_mapping(path: Path) -> dict[int, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    output: dict[int, np.ndarray] = {}
    with np.load(path) as payload:
        if "inds" not in payload or "data" not in payload:
            raise KeyError(f"{path} must contain 'inds' and 'data'")
        for source_index, value in zip(payload["inds"], payload["data"], strict=True):
            source_index = int(source_index)
            if source_index % SOURCE_FRAME_STRIDE:
                continue
            output[source_index // SOURCE_FRAME_STRIDE] = np.asarray(value)
    if not output:
        raise ValueError(f"{path} has no 0,{SOURCE_FRAME_STRIDE},... entries")
    return output


@dataclass(frozen=True)
class ActiveMemorySampleConfig:
    height: int = 480
    width: int = 832
    target_rgb_frames: int = 41
    query_blocks: int = 1
    query_block_gap: int = 16
    capture_span_min_frames: int = 161
    capture_span_max_frames: int = 1000
    curriculum_start_span: int = 257
    curriculum_epochs: int = 5
    candidate_views_min: int = 48
    candidate_views_max: int = 96
    episode_size: int = 8
    target_exclusion_radius: int = 2
    minimum_side_candidates: int = 8
    source_frame_stride: int = SOURCE_FRAME_STRIDE
    item_retries: int = 16

    def validate(self) -> None:
        if self.height % 16 or self.width % 16:
            raise ValueError("height and width must be divisible by 16")
        if self.target_rgb_frames < 2:
            raise ValueError("target_rgb_frames must be at least 2")
        if (self.target_rgb_frames - 1) % 4:
            raise ValueError("target_rgb_frames must equal 1 + 4k for the Wan VAE")
        if self.query_blocks < 1:
            raise ValueError("query_blocks must be positive")
        if self.query_block_gap < 0:
            raise ValueError("query_block_gap cannot be negative")
        if not 1 <= self.candidate_views_min <= self.candidate_views_max:
            raise ValueError("candidate view bounds are invalid")
        if self.episode_size < 1:
            raise ValueError("episode_size must be positive")
        if not (
            self.capture_span_min_frames
            <= self.curriculum_start_span
            <= self.capture_span_max_frames
        ):
            raise ValueError("curriculum_start_span must lie inside capture span bounds")
        if self.minimum_side_candidates < 1:
            raise ValueError("minimum_side_candidates must be positive")
        if self.source_frame_stride != SOURCE_FRAME_STRIDE:
            raise ValueError(
                f"this dataset is aligned to source stride {SOURCE_FRAME_STRIDE}"
            )
        if self.item_retries < 1:
            raise ValueError("item_retries must be positive")
        if self.capture_span_min_frames < self.minimum_required_frames:
            raise ValueError(
                "capture_span_min_frames is too short for target blocks and both capture sides"
            )

    @property
    def target_total_span(self) -> int:
        return self.query_blocks * self.target_rgb_frames + (
            self.query_blocks - 1
        ) * self.query_block_gap

    @property
    def minimum_required_frames(self) -> int:
        return self.target_total_span + 2 * (
            self.minimum_side_candidates + self.target_exclusion_radius
        )

    def capture_span_for_epoch(self, epoch: int) -> tuple[int, int]:
        if self.curriculum_epochs <= 1:
            maximum = self.capture_span_max_frames
        else:
            progress = min(max(epoch, 0) / (self.curriculum_epochs - 1), 1.0)
            maximum = round(
                self.curriculum_start_span
                + progress * (self.capture_span_max_frames - self.curriculum_start_span)
            )
        maximum = max(self.capture_span_min_frames, maximum)
        minimum = min(max(self.capture_span_min_frames, round(0.75 * maximum)), maximum)
        return minimum, maximum


@dataclass(frozen=True)
class ActiveMemorySample:
    item_name: str
    candidate_indices: tuple[int, ...]
    candidate_episode_ids: torch.Tensor
    candidate_rgb_uint8: torch.Tensor
    candidate_c2w: torch.Tensor
    candidate_intrinsics: torch.Tensor
    candidate_times: torch.Tensor
    target_index_blocks: tuple[tuple[int, ...], ...]
    target_rgb_uint8_blocks: tuple[torch.Tensor, ...]
    target_c2w_blocks: tuple[torch.Tensor, ...]
    target_intrinsics_blocks: tuple[torch.Tensor, ...]
    target_times_blocks: tuple[torch.Tensor, ...]
    image_hw: tuple[int, int]
    source_hw: tuple[int, int]
    capture_span: tuple[int, int]

    @property
    def num_candidates(self) -> int:
        return len(self.candidate_indices)


class VipeRoomTourItem:
    """Index one mounted room-tour scene without listing its RGB directory."""

    _RGB_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        artifact_candidates = (
            self.root / "vipe" / "vipe_artifacts",
            self.root / "vipe_artifacts",
        )
        self.artifact_root = next(
            (path for path in artifact_candidates if path.is_dir()), None
        )
        if self.artifact_root is None:
            raise FileNotFoundError(
                f"no vipe_artifacts below {self.root}; tried {artifact_candidates}"
            )
        self.rgb_root = self.root / "RGB"
        if not self.rgb_root.is_dir():
            raise FileNotFoundError(self.rgb_root)

        self.pose = _load_npz_mapping(self.artifact_root / "pose" / "video.npz")
        self.intrinsics = _load_npz_mapping(
            self.artifact_root / "intrinsics" / "video.npz"
        )
        self.indices = tuple(sorted(set(self.pose) & set(self.intrinsics)))
        if not self.indices:
            raise ValueError(f"{self.root.name} has no common pose/intrinsics indices")

        probe_index = self.indices[0] * SOURCE_FRAME_STRIDE
        self.rgb_suffix = next(
            (
                suffix
                for suffix in self._RGB_SUFFIXES
                if (self.rgb_root / f"{probe_index:06d}{suffix}").is_file()
            ),
            None,
        )
        if self.rgb_suffix is None:
            raise FileNotFoundError(
                f"cannot find RGB frame {probe_index:06d} with a supported suffix"
            )
        with Image.open(self.rgb_path(self.indices[0])) as image:
            self.source_hw = (image.height, image.width)

    def rgb_path(self, internal_index: int) -> Path:
        source_index = int(internal_index) * SOURCE_FRAME_STRIDE
        return self.rgb_root / f"{source_index:06d}{self.rgb_suffix}"

    def read_rgb_uint8(
        self,
        internal_index: int,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        path = self.rgb_path(internal_index)
        if not path.is_file():
            raise FileNotFoundError(
                f"missing sparse RGB frame for internal index {internal_index}: {path}"
            )
        target_h, target_w = target_hw
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.fit(
                image,
                (target_w, target_h),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
            array = np.asarray(image, dtype=np.uint8).copy()
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def read_video_uint8(
        self,
        indices: tuple[int, ...] | list[int],
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        return torch.stack(
            [self.read_rgb_uint8(index, target_hw) for index in indices], dim=1
        )

    def cameras(
        self,
        indices: tuple[int, ...] | list[int],
        target_hw: tuple[int, int],
        *,
        origin_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        c2w = torch.stack(
            [torch.from_numpy(self.pose[index]).float() for index in indices]
        )
        origin = torch.from_numpy(self.pose[origin_index]).float()
        c2w = normalize_c2w_to_first_capture(c2w, origin)
        intrinsics = torch.stack(
            [
                intrinsics_vector_to_matrix(
                    torch.from_numpy(self.intrinsics[index]).float()
                )
                for index in indices
            ]
        )
        intrinsics = resize_crop_intrinsics(intrinsics, self.source_hw, target_hw)
        return c2w, intrinsics

    @staticmethod
    def _stratified_sample(
        values: list[int],
        count: int,
        rng: random.Random,
    ) -> list[int]:
        if count >= len(values):
            return list(values)
        selected: list[int] = []
        for bin_index in range(count):
            left = bin_index * len(values) // count
            right = max(left + 1, (bin_index + 1) * len(values) // count)
            selected.append(values[rng.randrange(left, right)])
        return selected

    @staticmethod
    def _episode_ids(indices: list[int], episode_size: int) -> torch.Tensor:
        ids: list[int] = []
        episode = 0
        previous: int | None = None
        count = 0
        for index in indices:
            if previous is not None and (
                count >= episode_size
                or index - previous > max(4, episode_size * 4)
            ):
                episode += 1
                count = 0
            ids.append(episode)
            previous = index
            count += 1
        return torch.tensor(ids, dtype=torch.long)

    def sample(
        self,
        config: ActiveMemorySampleConfig,
        rng: random.Random,
        epoch: int,
        *,
        load_candidate_rgb: bool = True,
    ) -> ActiveMemorySample:
        config.validate()
        available = list(self.indices)
        span_min, span_max = config.capture_span_for_epoch(epoch)
        span_max = min(span_max, len(available))
        if span_max < span_min:
            raise RuntimeError(
                f"{self.root.name} has only {len(available)} aligned frames; "
                f"epoch {epoch} requires at least {span_min}"
            )
        span_length = rng.randint(span_min, span_max)
        span_offset = rng.randint(0, len(available) - span_length)
        span_indices = available[span_offset : span_offset + span_length]

        side = config.minimum_side_candidates + config.target_exclusion_radius
        target_start_min = side
        target_start_max = span_length - config.target_total_span - side
        if target_start_max < target_start_min:
            raise RuntimeError(f"{self.root.name} has no capture-pred-capture sample")
        target_offset = rng.randint(target_start_min, target_start_max)

        target_blocks: list[tuple[int, ...]] = []
        cursor = target_offset
        target_set: set[int] = set()
        for _ in range(config.query_blocks):
            block = tuple(span_indices[cursor : cursor + config.target_rgb_frames])
            if len(block) != config.target_rgb_frames:
                raise RuntimeError("incomplete target block")
            target_blocks.append(block)
            target_set.update(block)
            cursor += config.target_rgb_frames + config.query_block_gap

        first_target = target_blocks[0][0]
        last_target = target_blocks[-1][-1]
        excluded = {
            index
            for index in span_indices
            if any(
                abs(index - target_index) <= config.target_exclusion_radius
                for target_index in target_set
            )
        }
        before = [index for index in span_indices if index < first_target and index not in excluded]
        after = [index for index in span_indices if index > last_target and index not in excluded]
        middle = [
            index
            for index in span_indices
            if first_target <= index <= last_target
            and index not in excluded
            and index not in target_set
        ]
        if (
            len(before) < config.minimum_side_candidates
            or len(after) < config.minimum_side_candidates
        ):
            raise RuntimeError("capture-pred-capture sides are too short")

        desired = rng.randint(config.candidate_views_min, config.candidate_views_max)
        eligible = sorted(before + middle + after)
        desired = min(desired, len(eligible))
        side_count = min(config.minimum_side_candidates, desired // 3)
        selected = set(self._stratified_sample(before, side_count, rng))
        selected.update(self._stratified_sample(after, side_count, rng))
        remaining = [index for index in eligible if index not in selected]
        selected.update(
            self._stratified_sample(remaining, max(0, desired - len(selected)), rng)
        )
        candidate_indices = sorted(selected)
        if len(candidate_indices) < config.candidate_views_min:
            raise RuntimeError(
                f"only {len(candidate_indices)} valid candidate views, expected "
                f"at least {config.candidate_views_min}"
            )

        target_hw = (config.height, config.width)
        origin_index = candidate_indices[0]
        candidate_c2w, candidate_k = self.cameras(
            candidate_indices, target_hw, origin_index=origin_index
        )
        if load_candidate_rgb:
            candidate_rgb = self.read_video_uint8(
                candidate_indices, target_hw
            ).permute(1, 0, 2, 3)
        else:
            # Metadata-only mode is used by full-capture inference, which
            # builds a separate thumbnail bank and lazily reads only the views
            # selected by the agent.
            candidate_rgb = torch.empty(
                len(candidate_indices), 3, 0, 0, dtype=torch.uint8
            )
        candidate_times = torch.tensor(
            [index - origin_index for index in candidate_indices], dtype=torch.float32
        )

        target_rgb_blocks: list[torch.Tensor] = []
        target_c2w_blocks: list[torch.Tensor] = []
        target_k_blocks: list[torch.Tensor] = []
        target_times_blocks: list[torch.Tensor] = []
        for block in target_blocks:
            target_rgb_blocks.append(self.read_video_uint8(block, target_hw))
            c2w, intrinsics = self.cameras(block, target_hw, origin_index=origin_index)
            target_c2w_blocks.append(c2w)
            target_k_blocks.append(intrinsics)
            target_times_blocks.append(
                torch.tensor([index - origin_index for index in block], dtype=torch.float32)
            )

        return ActiveMemorySample(
            item_name=self.root.name,
            candidate_indices=tuple(candidate_indices),
            candidate_episode_ids=self._episode_ids(
                candidate_indices, config.episode_size
            ),
            candidate_rgb_uint8=candidate_rgb,
            candidate_c2w=candidate_c2w,
            candidate_intrinsics=candidate_k,
            candidate_times=candidate_times,
            target_index_blocks=tuple(target_blocks),
            target_rgb_uint8_blocks=tuple(target_rgb_blocks),
            target_c2w_blocks=tuple(target_c2w_blocks),
            target_intrinsics_blocks=tuple(target_k_blocks),
            target_times_blocks=tuple(target_times_blocks),
            image_hw=target_hw,
            source_hw=self.source_hw,
            capture_span=(span_indices[0], span_indices[-1]),
        )


class LocalRoomTourDataset(Dataset[ActiveMemorySample]):
    """One randomized scene sample per item and epoch from mounted storage."""

    def __init__(
        self,
        dataset_root: str | Path,
        config: ActiveMemorySampleConfig,
        *,
        item_names: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.root = Path(dataset_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        config.validate()
        self.config = config
        self.seed = int(seed)
        self.epoch = 0
        if item_names is None:
            item_names = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        cleaned = [Path(name.strip().rstrip("/")).name for name in item_names if name.strip()]
        self.total_items = len(cleaned)
        self.item_names = [
            name
            for name in cleaned
            if (
                estimate_sparse_frame_count(name) is None
                or estimate_sparse_frame_count(name) >= config.capture_span_min_frames
            )
        ]
        self.skipped_short_items = self.total_items - len(self.item_names)
        if not self.item_names:
            raise RuntimeError("no room-tour items satisfy the minimum capture span")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.item_names)

    def __getitem__(self, index: int) -> ActiveMemorySample:
        failures: list[str] = []
        attempts = min(self.config.item_retries, len(self.item_names))
        for attempt in range(attempts):
            candidate_index = (index + attempt) % len(self.item_names)
            item_name = self.item_names[candidate_index]
            rng = random.Random(
                self.seed
                + self.epoch * 1_000_003
                + index * 9_973
                + attempt * 104_729
            )
            try:
                item = VipeRoomTourItem(self.root / item_name)
                return item.sample(self.config, rng, self.epoch)
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                failures.append(f"{item_name}: {type(error).__name__}: {error}")
        joined = "\n  ".join(failures)
        raise RuntimeError(
            f"failed to obtain a valid room-tour sample after {attempts} attempts:\n  {joined}"
        )
