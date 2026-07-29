from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.geometry_aware_memory.data import (  # noqa: E402
    GeometryMemorySampleConfig,
    RcloneConfig,
    RoomTourItemCache,
    VipeRoomTourItem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item_name", required=True)
    parser.add_argument(
        "--dataset_root",
        default="h:bianyichen/AnyReconProDataset_labeled_2/",
    )
    parser.add_argument(
        "--cache_root",
        default="/tmp/lingbot_gim_world_cache",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_rgb_frames", type=int, default=81)
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
        "--break_after_download",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--rclone_binary", default="rclone")
    parser.add_argument("--rclone_config", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = RoomTourItemCache(
        args.dataset_root,
        args.cache_root,
        rclone=RcloneConfig(
            binary=args.rclone_binary,
            config_path=args.rclone_config,
        ),
    )
    print(f"Materializing {args.item_name} from {args.dataset_root}")
    local_root = cache.materialize(args.item_name)
    print(f"local_root={local_root}")
    if args.break_after_download:
        print("Breakpoint variables: local_root, cache, args")
        breakpoint()
    item = VipeRoomTourItem(local_root)
    config = GeometryMemorySampleConfig(
        height=args.height,
        width=args.width,
        target_rgb_frames=args.target_rgb_frames,
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
