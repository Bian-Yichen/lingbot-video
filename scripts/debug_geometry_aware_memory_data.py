from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.geometry_aware_memory.data import (  # noqa: E402
    SOURCE_FRAME_STRIDE,
    GeometryMemorySampleConfig,
    LocalRoomTourIndex,
    VipeRoomTourItem,
)


def _config_defaults() -> dict[str, Any]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    known, _ = bootstrap.parse_known_args()
    if not known.config:
        return {}
    payload = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("debug config must be one JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one local target and retrieved-memory sample."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--item_name", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_rgb_frames", type=int, default=41)
    parser.add_argument("--query_blocks", type=int, default=1)
    parser.add_argument("--local_window_rgb_frames", type=int, default=81)
    parser.add_argument("--memory_views_min", type=int, default=2)
    parser.add_argument("--memory_views_max", type=int, default=24)
    parser.add_argument("--retrieval_rotation_weight", type=float, default=0.25)
    parser.add_argument("--retrieval_temperature", type=float, default=0.25)
    parser.add_argument("--sample_epoch", type=int, default=0)
    parser.add_argument("--local_window_start", type=int, default=None)
    parser.add_argument("--target_start", type=int, default=None)
    parser.add_argument("--memory_view_count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--break_after_resolve",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--break_after_sample",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    if not args.item_name:
        parser.error("--item_name is required (it may be supplied by --config)")
    if not args.dataset_root:
        parser.error("--dataset_root is required (it may be supplied by --config)")
    explicit = (
        args.local_window_start,
        args.target_start,
        args.memory_view_count,
    )
    if any(value is not None for value in explicit) and not all(
        value is not None for value in explicit
    ):
        parser.error(
            "explicit sampling requires local_window_start, target_start, "
            "and memory_view_count"
        )
    return args


def main() -> None:
    args = parse_args()
    print(f"Resolving {args.item_name} below {args.dataset_root}")
    item_index = LocalRoomTourIndex(args.dataset_root)
    local_root = item_index.item_path(args.item_name)
    print(f"local_root={local_root}")
    if args.break_after_resolve:
        print("Breakpoint variables: local_root, item_index, args")
        breakpoint()

    item = VipeRoomTourItem(local_root)
    config = GeometryMemorySampleConfig(
        height=args.height,
        width=args.width,
        target_rgb_frames=args.target_rgb_frames,
        query_blocks=args.query_blocks,
        local_window_rgb_frames=args.local_window_rgb_frames,
        memory_views_min=args.memory_views_min,
        memory_views_max=args.memory_views_max,
        retrieval_rotation_weight=args.retrieval_rotation_weight,
        retrieval_temperature=args.retrieval_temperature,
    )
    sample = item.make_sample(
        config,
        random.Random(args.seed),
        epoch=args.sample_epoch,
        local_window_start=args.local_window_start,
        target_start=args.target_start,
        memory_view_count=args.memory_view_count,
    )
    capture_set = set(sample.capture_rgb_indices)
    query_indices = tuple(
        index
        for block in sample.query_rgb_blocks
        for index in block
    )
    query_set = set(query_indices)
    query_deltas = [
        right - left
        for left, right in zip(query_indices, query_indices[1:])
    ]
    capture_c2w, _ = item.cameras(
        [sample.capture_start],
        sample.image_hw,
        origin_index=sample.capture_start,
    )
    report = {
        "item": item.root.name,
        "local_root": str(item.root),
        "source_hw": item.source_hw,
        "internal_common_frames": len(item.indices),
        "internal_range": [item.indices[0], item.indices[-1]],
        "source_index_mapping": (
            f"internal i corresponds to source index {SOURCE_FRAME_STRIDE}*i"
        ),
        "sample_epoch": args.sample_epoch,
        "local_window_internal_range": [
            sample.local_window_start,
            sample.local_window_end,
        ],
        "local_window_source_range": [
            SOURCE_FRAME_STRIDE * sample.local_window_start,
            SOURCE_FRAME_STRIDE * sample.local_window_end,
        ],
        "local_window_rgb_frames": (
            sample.local_window_end - sample.local_window_start + 1
        ),
        "memory_view_internal_indices": list(sample.capture_rgb_indices),
        "memory_view_source_indices": [
            SOURCE_FRAME_STRIDE * index
            for index in sample.capture_rgb_indices
        ],
        "memory_rgb_frames": len(sample.capture_rgb_indices),
        "memory_latent_frames": len(sample.capture_rgb_indices),
        "query_blocks_internal_ranges": [
            [block[0], block[-1]] for block in sample.query_rgb_blocks
        ],
        "query_blocks_source_ranges": [
            [
                SOURCE_FRAME_STRIDE * block[0],
                SOURCE_FRAME_STRIDE * block[-1],
            ]
            for block in sample.query_rgb_blocks
        ],
        "query_rgb_frames_total": len(query_indices),
        "query_contiguous": set(query_deltas) == {1},
        "query_latent_frames_per_block": config.target_latent_frames,
        "geometry_query_internal_indices": list(
            sample.geometry_query_indices
        ),
        "capture_query_disjoint": capture_set.isdisjoint(query_set),
        "query_inside_local_window": (
            query_indices[0] >= sample.local_window_start
            and query_indices[-1] <= sample.local_window_end
        ),
        "retrieval_coverage_score_higher_is_better": (
            sample.retrieval_coverage_score
        ),
        "trajectory_overlap_score_lower_is_better": (
            sample.trajectory_overlap_score
        ),
        "capture_first_pose_is_identity_after_normalization": (
            capture_c2w[0].tolist()
        ),
        "cache_mode": "disabled; VAE and VGGT run online during training",
        "iteration_semantics": (
            "one scene once per epoch; epoch changes the local window, memory "
            "view count, and pose-retrieval result"
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.break_after_sample:
        print("Breakpoint variables: item, config, sample, report")
        breakpoint()


if __name__ == "__main__":
    main()
