from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
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
        description=(
            "Debug one Latent Spatial Memory dataset item in the main process. "
            "This script does not load LingBot, Accelerate, or a DataLoader."
        )
    )
    parser.add_argument("--item_name", required=True)
    parser.add_argument(
        "--dataset_root",
        default="/data/bianyichen/H-hdd/AnyReconProDataset_labeled_2",
    )
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
    parser.add_argument(
        "--tree_limit",
        type=int,
        default=200,
        help="Maximum number of local files to print; 0 disables the limit.",
    )
    parser.add_argument(
        "--break_at",
        choices=("none", "after_resolve", "after_index"),
        default="none",
        help="Enter an interactive breakpoint in this main-process-only script.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Decode one complete RGB-D training sample after the cheap index audit.",
    )
    return parser.parse_args()


def _format_bytes(size: int) -> str:
    value = float(size)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _print_tree(root: Path, limit: int) -> None:
    print("\n=== LOCAL FILES ===")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    visible = files if limit <= 0 else files[:limit]
    for path in visible:
        print(f"{path.relative_to(root)}  ({_format_bytes(path.stat().st_size)})")
    if len(visible) < len(files):
        print(f"... omitted {len(files) - len(visible)} files; use --tree_limit 0 for all")
    print(f"total files: {len(files)}")


def _print_expected_layout(root: Path) -> None:
    artifact_root = root / "vipe" / "vipe_artifacts"
    important = (
        root / "RGB",
        root / "chunk_metadata.json",
        artifact_root / "pose" / "video.npz",
        artifact_root / "intrinsics" / "video.npz",
        artifact_root / "intrinsics" / "video_camera.txt",
        artifact_root / "depth" / "video.zip",
        artifact_root / "depth" / "video.zip.parts",
    )
    print("\n=== EXPECTED LAYOUT ===")
    for path in important:
        if path.is_dir():
            state = "directory"
        elif path.is_file():
            state = f"file, {_format_bytes(path.stat().st_size)}"
        else:
            state = "MISSING"
        print(f"{path.relative_to(root)!s:65s} {state}")

    rgb_root = root / "RGB"
    if rgb_root.is_dir():
        rgb_files = sum(1 for path in rgb_root.iterdir() if path.is_file())
        print(f"RGB direct child files: {rgb_files}")

    depth_parts = artifact_root / "depth" / "video.zip.parts"
    if depth_parts.is_dir():
        complete = sorted(depth_parts.glob("*.zip"))
        partial = sorted(depth_parts.glob("*.partial"))
        print(f"complete depth shards: {[path.name for path in complete]}")
        print(f"partial depth shards:  {[path.name for path in partial]}")


def _contiguous_run_summary(indices: set[int]) -> tuple[int, int]:
    if not indices:
        return 0, 0
    ordered = sorted(indices)
    runs = 1
    longest = 1
    current = 1
    for previous, value in zip(ordered, ordered[1:]):
        if value == previous + 1:
            current += 1
            longest = max(longest, current)
        else:
            runs += 1
            current = 1
    return runs, longest


def _print_index_summary(name: str, indices: set[int]) -> None:
    if not indices:
        print(f"{name:12s} count=0")
        return
    ordered = sorted(indices)
    runs, longest = _contiguous_run_summary(indices)
    print(
        f"{name:12s} count={len(indices):6d} "
        f"range={ordered[0]}..{ordered[-1]} "
        f"runs={runs} longest_contiguous={longest} "
        f"first={ordered[:8]} last={ordered[-8:]}"
    )


def _audit_target_candidates(
    item: VipeRoomTourItem,
    config: LongTrajectorySampleConfig,
) -> list[int]:
    config.validate()
    rgb = set(item.rgb_by_index)
    pose = set(item.pose_by_index)
    intrinsics = set(item.intrinsics_by_index)
    depth = set(item.depth_location)
    geometry = pose & intrinsics & depth
    available = sorted(rgb & geometry)

    print("\n=== MODALITY INDICES ===")
    print(
        "Sparse source index s -> internal training index "
        f"s / {SOURCE_FRAME_STRIDE}; source indices must be divisible by "
        f"{SOURCE_FRAME_STRIDE}"
    )
    modality_source_indices = {
        "RGB": item.rgb_source_index_by_index,
        "depth": item.depth_source_index_by_index,
        "pose": item.pose_source_index_by_index,
        "intrinsics": item.intrinsics_source_index_by_index,
    }
    for name, source_mapping in modality_source_indices.items():
        examples = [
            (internal, source_mapping[internal])
            for internal in sorted(source_mapping)[:12]
        ]
        print(f"{name} mapping examples (internal, source): {examples}")
    _print_index_summary("RGB", rgb)
    _print_index_summary("depth", depth)
    _print_index_summary("pose", pose)
    _print_index_summary("intrinsics", intrinsics)
    _print_index_summary("geometry", geometry)
    _print_index_summary("all-common", set(available))

    if not available:
        print("\nNo common frame index exists across RGB/depth/pose/intrinsics.")
        first_indices = {
            name: min(values) if values else None
            for name, values in {
                "RGB": rgb,
                "depth": depth,
                "pose": pose,
                "intrinsics": intrinsics,
            }.items()
        }
        print(f"first-index comparison: {first_indices}")
        return []

    target_rgb_offsets = tuple(range(config.target_rgb_frames))
    target_latent_offsets = tuple(
        range(0, config.target_rgb_frames, config.vae_temporal_stride)
    )
    preceding_rgb_offsets = tuple(range(-config.preceding_rgb_frames, 0))
    preceding_latent_offsets = tuple(
        range(
            -config.preceding_rgb_frames,
            0,
            config.vae_temporal_stride,
        )
    )
    final_capture_start_offset = 1 - config.capture_clip_rgb_frames
    final_capture_rgb_offsets = tuple(range(final_capture_start_offset, 1))
    capture_anchor_offsets = tuple(
        range(
            final_capture_start_offset,
            1,
            config.vae_temporal_stride,
        )
    )

    rejected: Counter[str] = Counter()
    examples: dict[str, tuple[int, list[int]]] = {}
    candidates: list[int] = []

    def reject(reason: str, start: int, missing: list[int]) -> None:
        rejected[reason] += 1
        examples.setdefault(reason, (start, missing[:16]))

    for start in available:
        if start - available[0] < config.history_min_frames:
            reject(
                "history shorter than history_min_frames",
                start,
                [config.history_min_frames - (start - available[0])],
            )
            continue
        missing = [start + offset for offset in target_rgb_offsets if start + offset not in rgb]
        if missing:
            reject("missing target RGB", start, missing)
            continue
        missing = [
            start + offset
            for offset in preceding_rgb_offsets
            if start + offset not in rgb
        ]
        if missing:
            reject("missing preceding RGB", start, missing)
            continue
        missing = [
            start + offset
            for offset in target_latent_offsets
            if start + offset not in geometry
        ]
        if missing:
            reject("missing target geometry", start, missing)
            continue
        missing = [
            start + offset
            for offset in preceding_latent_offsets
            if start + offset not in geometry
        ]
        if missing:
            reject("missing preceding geometry", start, missing)
            continue
        missing = [
            start + offset
            for offset in final_capture_rgb_offsets
            if start + offset not in rgb
        ]
        if missing:
            reject("incomplete overlap capture RGB", start, missing)
            continue
        missing = [
            start + offset
            for offset in capture_anchor_offsets
            if start + offset not in geometry
        ]
        if missing:
            reject("incomplete overlap capture geometry", start, missing)
            continue
        candidates.append(start)

    print("\n=== TARGET WINDOW AUDIT ===")
    print(
        f"config: history={config.history_min_frames}..{config.history_max_frames}, "
        f"target_rgb={config.target_rgb_frames}, "
        f"capture_clips={config.capture_clips}, "
        f"capture_rgb_per_clip={config.capture_clip_rgb_frames}"
    )
    for reason, count in rejected.most_common():
        start, missing = examples[reason]
        if reason == "history shorter than history_min_frames":
            detail = f"first example start={start}, missing_history_span={missing[0]}"
        else:
            detail = f"first example start={start}, missing_indices={missing}"
        print(f"rejected {count:6d}: {reason}; {detail}")
    if candidates:
        print(
            f"valid target starts: {len(candidates)}, "
            f"range={candidates[0]}..{candidates[-1]}, "
            f"first={candidates[:16]}"
        )
    else:
        print("valid target starts: 0")

    implementation_candidates = item._candidate_target_starts(config)
    if candidates != implementation_candidates:
        raise AssertionError(
            "debug audit and VipeRoomTourItem._candidate_target_starts disagree"
        )
    return candidates


def _print_depth_archives(item: VipeRoomTourItem) -> None:
    archive_counts: Counter[Path] = Counter(
        archive for archive, _member in item.depth_location.values()
    )
    print("\n=== COMPLETE DEPTH ARCHIVES USED ===")
    for archive, count in sorted(archive_counts.items(), key=lambda pair: str(pair[0])):
        print(f"{archive.relative_to(item.root)}: {count} indexed EXR frames")


def _print_sample(sample: dict[str, torch.Tensor | str]) -> None:
    print("\n=== DECODED SAMPLE ===")
    for key, value in sample.items():
        if torch.is_tensor(value):
            finite = (
                float(torch.isfinite(value).float().mean())
                if value.is_floating_point()
                else 1.0
            )
            print(
                f"{key:28s} shape={tuple(value.shape)!s:24s} "
                f"dtype={str(value.dtype):14s} finite={finite:.6f}"
            )
        else:
            print(f"{key:28s} {value}")


def main() -> None:
    args = parse_args()
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
    item_index = LocalRoomTourIndex(args.dataset_root)

    print("=== RESOLVE LOCAL SCENE ===")
    print(f"dataset root: {args.dataset_root}")
    print(f"item:         {args.item_name}")
    local_root = item_index.item_path(args.item_name)
    print(f"local root:   {local_root}")
    _print_expected_layout(local_root)
    _print_tree(local_root, args.tree_limit)

    if args.break_at == "after_resolve":
        print("\nBreakpoint after resolve. Useful variables: local_root, config, args")
        breakpoint()

    print("\n=== INDEX LOCAL ITEM ===")
    # Deliberately do not catch this exception. A malformed local layout should
    # terminate here with the original traceback rather than be mistaken for a
    # bad item inside an infinite IterableDataset.
    item = VipeRoomTourItem(local_root)
    print(f"source image HxW: {item.source_hw}")
    _print_depth_archives(item)
    candidates = _audit_target_candidates(item, config)

    if args.break_at == "after_index":
        print(
            "\nBreakpoint after indexing. Useful variables: "
            "item, candidates, local_root, config"
        )
        breakpoint()

    if not candidates:
        raise RuntimeError(
            "No valid target window. See TARGET WINDOW AUDIT above for the exact "
            "rejection counts. No model or DataLoader was started."
        )

    if args.sample:
        sample = item.sample(config, random.Random(args.seed))
        _print_sample(sample)
    else:
        print("\nIndex audit passed. Add --sample to decode one complete training sample.")


if __name__ == "__main__":
    main()
