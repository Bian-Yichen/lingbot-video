from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.active_world_memory.data import (  # noqa: E402
    ActiveMemorySampleConfig,
    VipeRoomTourItem,
)


def _config(cls, values):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _contact_sheet(video, labels, columns=8, thumb_hw=(135, 234)):
    height, width = thumb_hw
    rows = (video.shape[0] + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * (height + 22)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(video):
        image = Image.fromarray(frame.transpose(1, 2, 0)).resize((width, height))
        x = (index % columns) * width
        y = (index // columns) * (height + 22)
        sheet.paste(image, (x, y))
        draw.text((x + 4, y + height + 3), str(labels[index]), fill="white")
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--item_name", required=True)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="debug_active_world_memory")
    args = parser.parse_args()

    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = _config(ActiveMemorySampleConfig, payload.get("data", {}))
    item = VipeRoomTourItem(Path(payload["dataset_root"]) / args.item_name)
    sample = item.sample(config, random.Random(args.seed), args.epoch)

    print(f"item: {sample.item_name}")
    print(f"source HxW: {sample.source_hw}; training HxW: {sample.image_hw}")
    print(f"aligned item frames: {len(item.indices)} ({item.indices[0]}..{item.indices[-1]})")
    print(f"capture span: {sample.capture_span}")
    print(f"candidate views: {sample.num_candidates}")
    print(f"candidate indices: {sample.candidate_indices}")
    print(f"episodes: {int(sample.candidate_episode_ids.max().item()) + 1}")
    for block, indices in enumerate(sample.target_index_blocks):
        print(f"target block {block}: {indices[0]}..{indices[-1]} ({len(indices)} frames)")
    overlap = set(sample.candidate_indices) & {
        index for block in sample.target_index_blocks for index in block
    }
    print(f"candidate/target exact overlap: {sorted(overlap)}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate = sample.candidate_rgb_uint8.numpy()
    _contact_sheet(candidate, sample.candidate_indices).save(
        output / "candidate_memory_views.jpg", quality=92
    )
    for block_index, (rgb, indices) in enumerate(
        zip(sample.target_rgb_uint8_blocks, sample.target_index_blocks, strict=True)
    ):
        _contact_sheet(rgb.permute(1, 0, 2, 3).numpy(), indices).save(
            output / f"target_block_{block_index:02d}.jpg", quality=92
        )
    np.savez_compressed(
        output / "camera_sample.npz",
        candidate_indices=np.asarray(sample.candidate_indices),
        candidate_episode_ids=sample.candidate_episode_ids.numpy(),
        candidate_c2w=sample.candidate_c2w.numpy(),
        candidate_intrinsics=sample.candidate_intrinsics.numpy(),
        target_indices=np.asarray(sample.target_index_blocks),
        target_c2w=np.stack([value.numpy() for value in sample.target_c2w_blocks]),
        target_intrinsics=np.stack(
            [value.numpy() for value in sample.target_intrinsics_blocks]
        ),
    )
    print(f"saved audit artifacts to {output.resolve()}")


if __name__ == "__main__":
    main()
