from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.latent_spatial_memory.data import (  # noqa: E402
    LongTrajectorySampleConfig,
    RcloneConfig,
    RoomTourItemCache,
    VipeRoomTourItem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize one remote VIPE item and validate a training sample."
    )
    parser.add_argument(
        "--dataset_root",
        default="h:bianyichen/AnyReconProDataset_labeled/",
    )
    parser.add_argument("--cache_root", default="/tmp/lingbot_latent_memory_cache")
    parser.add_argument("--item_name", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--capture_frames", type=int, default=48)
    parser.add_argument("--history_min_frames", type=int, default=256)
    parser.add_argument("--history_max_frames", type=int, default=4096)
    parser.add_argument("--rollout_chunks", type=int, default=2)
    parser.add_argument("--latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rclone_config", default=None)
    parser.add_argument(
        "--rclone_clear_proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = RoomTourItemCache(
        args.dataset_root,
        args.cache_root,
        rclone=RcloneConfig(
            config_path=args.rclone_config,
            clear_proxy=args.rclone_clear_proxy,
        ),
    )
    item_name = args.item_name or cache.list_items()[0]
    item = VipeRoomTourItem(cache.materialize(item_name))
    config = LongTrajectorySampleConfig(
        height=args.height,
        width=args.width,
        capture_frames=args.capture_frames,
        history_min_frames=args.history_min_frames,
        history_max_frames=args.history_max_frames,
        rollout_chunks=args.rollout_chunks,
        latent_frames_per_chunk=args.latent_frames_per_chunk,
    )
    sample = item.sample(config, random.Random(args.seed))
    print(f"item={item_name}")
    for key, value in sample.items():
        if torch.is_tensor(value):
            finite = (
                float(torch.isfinite(value).float().mean())
                if value.is_floating_point()
                else 1.0
            )
            print(
                f"{key}: shape={tuple(value.shape)} dtype={value.dtype} "
                f"finite={finite:.6f}"
            )
        else:
            print(f"{key}: {value}")
    identity_error = (
        sample["capture_c2w"][0] - torch.eye(4)
    ).abs().max()
    print(f"first_capture_identity_max_error={float(identity_error):.6g}")
    print(
        "capture_span="
        f"{int(sample['capture_indices'][0])}..{int(sample['capture_indices'][-1])} "
        f"target_span={int(sample['target_rgb_indices'][0])}.."
        f"{int(sample['target_rgb_indices'][-1])}"
    )


if __name__ == "__main__":
    main()
