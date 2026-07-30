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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--item_name", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_rgb_frames", type=int, default=81)
    parser.add_argument("--vae_temporal_stride", type=int, default=4)
    parser.add_argument("--target_start", type=int, default=None)
    parser.add_argument("--min_memory_rgb_frames", type=int, default=800)
    parser.add_argument("--target_guard_rgb_frames", type=int, default=128)
    parser.add_argument(
        "--context_policy",
        choices=["all_except_target", "prefix"],
        default="prefix",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--break_after_resolve",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    if not args.item_name:
        parser.error("--item_name is required (it may be supplied by --config)")
    if not args.dataset_root:
        parser.error("--dataset_root is required (it may be supplied by --config)")
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
        vae_temporal_stride=args.vae_temporal_stride,
        min_memory_rgb_frames=args.min_memory_rgb_frames,
        target_guard_rgb_frames=args.target_guard_rgb_frames,
        samples_per_item=1,
        context_policy=args.context_policy,
    )
    starts = item.valid_target_starts(config)
    sample = item.make_sample(
        config,
        random.Random(args.seed),
        target_start=args.target_start,
    )
    report = {
        "item": item.root.name,
        "source_hw": item.source_hw,
        "internal_common_frames": len(item.indices),
        "internal_range": [item.indices[0], item.indices[-1]],
        "source_index_mapping": (
            "internal i corresponds to source RGB/pose/intrinsics index 5*i"
        ),
        "valid_target_starts": len(starts),
        "valid_target_range": [starts[0], starts[-1]],
        "sample_target_start": sample.target_start,
        "sample_target_rgb_count": len(sample.target_rgb_indices),
        "sample_memory_rgb_count": len(sample.memory_rgb_indices),
        "sample_geometry_query_index": sample.geometry_query_index,
        "expected_full_scene_vae_latents": (
            1 + (len(item.indices) - 1) // config.vae_temporal_stride
        ),
        "expected_target_vae_latents": config.target_latent_frames,
        "expected_memory_candidates_after_vae": sum(
            index in set(sample.memory_rgb_indices)
            for index in item.indices[:: config.vae_temporal_stride]
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
