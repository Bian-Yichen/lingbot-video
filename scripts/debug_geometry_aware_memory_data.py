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
        description="Audit one continuous-capture/independent-query sample."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--item_name", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_rgb_frames", type=int, default=81)
    parser.add_argument("--query_blocks", type=int, default=2)
    parser.add_argument("--vae_temporal_stride", type=int, default=4)
    parser.add_argument("--capture_min_rgb_frames", type=int, default=257)
    parser.add_argument("--capture_max_rgb_frames", type=int, default=801)
    parser.add_argument(
        "--capture_curriculum_start_max_rgb_frames",
        type=int,
        default=321,
    )
    parser.add_argument("--capture_curriculum_epochs", type=int, default=5)
    parser.add_argument(
        "--capture_min_fraction_of_current_max",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--capture_query_guard_rgb_frames",
        type=int,
        default=32,
    )
    parser.add_argument("--trajectory_candidate_trials", type=int, default=128)
    parser.add_argument("--trajectory_topk", type=int, default=8)
    parser.add_argument("--trajectory_pose_stride", type=int, default=4)
    parser.add_argument("--trajectory_rotation_weight", type=float, default=0.25)
    parser.add_argument("--sample_epoch", type=int, default=0)
    parser.add_argument("--capture_start", type=int, default=None)
    parser.add_argument("--query_start", type=int, default=None)
    parser.add_argument("--capture_rgb_frames", type=int, default=None)
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
    if (args.capture_start is None) != (args.query_start is None):
        parser.error("--capture_start and --query_start must be set together")
    if args.capture_rgb_frames is not None and args.capture_start is None:
        parser.error(
            "--capture_rgb_frames requires explicit capture/query starts"
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
        vae_temporal_stride=args.vae_temporal_stride,
        capture_min_rgb_frames=args.capture_min_rgb_frames,
        capture_max_rgb_frames=args.capture_max_rgb_frames,
        capture_curriculum_start_max_rgb_frames=(
            args.capture_curriculum_start_max_rgb_frames
        ),
        capture_curriculum_epochs=args.capture_curriculum_epochs,
        capture_min_fraction_of_current_max=(
            args.capture_min_fraction_of_current_max
        ),
        capture_query_guard_rgb_frames=args.capture_query_guard_rgb_frames,
        trajectory_candidate_trials=args.trajectory_candidate_trials,
        trajectory_topk=args.trajectory_topk,
        trajectory_pose_stride=args.trajectory_pose_stride,
        trajectory_rotation_weight=args.trajectory_rotation_weight,
    )
    sample = item.make_sample(
        config,
        random.Random(args.seed),
        epoch=args.sample_epoch,
        capture_start=args.capture_start,
        query_start=args.query_start,
        capture_rgb_frames=args.capture_rgb_frames,
    )
    capture_set = set(sample.capture_rgb_indices)
    query_indices = tuple(
        index
        for block in sample.query_rgb_blocks
        for index in block
    )
    query_set = set(query_indices)
    temporal_gap = max(
        query_indices[0] - sample.capture_rgb_indices[-1] - 1,
        sample.capture_rgb_indices[0] - query_indices[-1] - 1,
    )
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
        "capture_curriculum_rgb_bounds": config.capture_bounds_for_epoch(
            args.sample_epoch
        ),
        "capture_internal_range": [
            sample.capture_rgb_indices[0],
            sample.capture_rgb_indices[-1],
        ],
        "capture_source_range": [
            SOURCE_FRAME_STRIDE * sample.capture_rgb_indices[0],
            SOURCE_FRAME_STRIDE * sample.capture_rgb_indices[-1],
        ],
        "capture_rgb_frames": len(sample.capture_rgb_indices),
        "capture_latent_frames": (
            1
            + (len(sample.capture_rgb_indices) - 1)
            // config.vae_temporal_stride
        ),
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
        "query_latent_frames_per_block": config.target_latent_frames,
        "geometry_query_internal_indices": list(
            sample.geometry_query_indices
        ),
        "capture_query_disjoint": capture_set.isdisjoint(query_set),
        "capture_query_temporal_gap": temporal_gap,
        "required_temporal_guard": config.capture_query_guard_rgb_frames,
        "trajectory_overlap_score_lower_is_better": (
            sample.trajectory_overlap_score
        ),
        "capture_first_pose_is_identity_after_normalization": (
            capture_c2w[0].tolist()
        ),
        "cache_mode": "disabled; VAE and VGGT run online during training",
        "iteration_semantics": (
            "one scene once per epoch; epoch changes the random trajectory pair"
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.break_after_sample:
        print("Breakpoint variables: item, config, sample, report")
        breakpoint()


if __name__ == "__main__":
    main()
