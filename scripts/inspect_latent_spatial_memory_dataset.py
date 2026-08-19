from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.latent_spatial_memory.data import (  # noqa: E402
    LocalRoomTourIndex,
    SOURCE_FRAME_STRIDE,
    LongTrajectorySampleConfig,
    VipeRoomTourItem,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open one local VIPE item and validate a training sample."
    )
    parser.add_argument(
        "--dataset_root",
        default="/data/bianyichen/H-hdd/AnyReconProDataset_labeled_2",
    )
    parser.add_argument("--item_name", default=None)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--capture_clips", type=int, default=16)
    parser.add_argument("--capture_clip_rgb_frames", type=int, default=9)
    parser.add_argument("--preceding_rgb_frames", type=int, default=8)
    parser.add_argument("--reference_frames", type=int, default=4)
    parser.add_argument("--history_min_frames", type=int, default=256)
    parser.add_argument("--history_max_frames", type=int, default=4096)
    parser.add_argument("--latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    item_index = LocalRoomTourIndex(args.dataset_root)
    item_name = args.item_name or item_index.list_items()[0]
    item = VipeRoomTourItem(item_index.item_path(item_name))
    print(
        "Sparse RGB/depth/pose/intrinsics source index s -> internal index "
        f"s / {SOURCE_FRAME_STRIDE}"
    )
    for name, source_mapping in {
        "RGB": item.rgb_source_index_by_index,
        "depth": item.depth_source_index_by_index,
        "pose": item.pose_source_index_by_index,
        "intrinsics": item.intrinsics_source_index_by_index,
    }.items():
        print(
            f"{name} mapping examples:",
            [
                (internal, source_mapping[internal])
                for internal in sorted(source_mapping)[:12]
            ],
        )
    config = LongTrajectorySampleConfig(
        height=args.height,
        width=args.width,
        capture_clips=args.capture_clips,
        capture_clip_rgb_frames=args.capture_clip_rgb_frames,
        preceding_rgb_frames=args.preceding_rgb_frames,
        reference_frames=args.reference_frames,
        history_min_frames=args.history_min_frames,
        history_max_frames=args.history_max_frames,
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
