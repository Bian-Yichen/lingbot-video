from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.geometry_aware_memory.data import (  # noqa: E402
    GeometryMemorySampleConfig,
    VipeRoomTourItem,
)
from lingbot_video.geometry_aware_memory.memory_reconstruction import (  # noqa: E402
    GIMMemoryReconstructionModel,
    MemoryReconstructionConfig,
    load_lingbot_patch_embedder,
)
from lingbot_video.geometry_aware_memory.reconstruction_training import (  # noqa: E402
    prepare_reconstruction_batch,
)


def _config_defaults() -> dict[str, Any]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    known, _ = bootstrap.parse_known_args()
    if not known.config:
        return {}
    payload = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("inference config must be one JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode held-out RGB views from a trained GIM memory checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--item_name", required=True)
    parser.add_argument("--output_dir", default="outputs/memory_reconstruction_eval")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_rgb_frames", type=int, default=2)
    parser.add_argument("--query_blocks", type=int, default=1)
    parser.add_argument("--local_window_rgb_frames", type=int, default=81)
    parser.add_argument("--memory_views_min", type=int, default=12)
    parser.add_argument("--memory_views_max", type=int, default=32)
    parser.add_argument("--retrieval_rotation_weight", type=float, default=0.25)
    parser.add_argument("--retrieval_temperature", type=float, default=0.25)
    parser.add_argument("--vae_encode_chunk_rgb_frames", type=int, default=16)
    parser.add_argument("--local_window_start", type=int, default=None)
    parser.add_argument("--target_start", type=int, default=None)
    parser.add_argument("--memory_view_count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    if not args.model_dir or not args.dataset_root:
        parser.error("model_dir and dataset_root are required")
    explicit = (
        args.local_window_start,
        args.target_start,
        args.memory_view_count,
    )
    if any(value is not None for value in explicit) and not all(
        value is not None for value in explicit
    ):
        parser.error(
            "local_window_start, target_start and memory_view_count must be "
            "provided together"
        )
    return args


def _dtype(name: str) -> torch.dtype:
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _checkpoint_file(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "memory_reconstruction.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _normalize_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    for name in (
        "latent_patch_size",
        "encoder_axes_dims",
        "encoder_axes_lens",
    ):
        if name in output:
            output[name] = tuple(output[name])
    return output


def _save_rgb(path: Path, tensor: torch.Tensor) -> None:
    array = (
        tensor.detach()
        .float()
        .clamp(0, 1)
        .mul(255)
        .round()
        .byte()
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(np.asarray(array)).save(path)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("memory reconstruction inference requires CUDA")
    device = torch.device("cuda")
    compute_dtype = _dtype(args.mixed_precision)
    sample_config = GeometryMemorySampleConfig(
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
    item = VipeRoomTourItem(Path(args.dataset_root) / args.item_name)
    sample = item.make_sample(
        sample_config,
        random.Random(args.seed),
        local_window_start=args.local_window_start,
        target_start=args.target_start,
        memory_view_count=args.memory_view_count,
    )
    sample = item.preload_sample(sample)

    checkpoint_payload = torch.load(
        _checkpoint_file(args.checkpoint),
        map_location="cpu",
        weights_only=False,
    )
    patch_config, patch_embedder = load_lingbot_patch_embedder(args.model_dir)
    model_config = MemoryReconstructionConfig(
        **_normalize_model_config(checkpoint_payload["model_config"])
    )
    if (
        model_config.encoder_hidden_size != patch_config.hidden_size
        or model_config.latent_patch_size != patch_config.patch_size
    ):
        raise ValueError("checkpoint model config does not match LingBot patch weights")
    model = GIMMemoryReconstructionModel(patch_embedder, model_config)
    model.load_state_dict(checkpoint_payload["model"], strict=True)
    model.eval().requires_grad_(False).to(device)
    vae = AutoencoderKLWan.from_pretrained(
        args.model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval().requires_grad_(False).to(device)
    prepared = prepare_reconstruction_batch(
        sample,
        vae=vae,
        vae_encode_chunk_rgb_frames=args.vae_encode_chunk_rgb_frames,
        device=device,
        compute_dtype=compute_dtype,
    )
    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=compute_dtype,
        enabled=compute_dtype != torch.float32,
    ):
        prediction, memory = model(
            prepared.history_latents.to(device=device, dtype=compute_dtype),
            prepared.history_c2w.to(device),
            prepared.history_intrinsics.to(device),
            prepared.target_c2w.to(device),
            prepared.target_intrinsics.to(device),
        )

    output_dir = Path(args.output_dir) / args.item_name
    output_dir.mkdir(parents=True, exist_ok=True)
    target_indices = [
        index for block in sample.query_rgb_blocks for index in block
    ]
    for view_index, frame_index in enumerate(target_indices):
        _save_rgb(
            output_dir / f"frame-{frame_index:06d}-prediction.png",
            prediction[0, view_index],
        )
        _save_rgb(
            output_dir / f"frame-{frame_index:06d}-target.png",
            prepared.target_rgb[0, view_index],
        )
    mse = torch.mean(
        (prediction.detach().float().cpu() - prepared.target_rgb.float()) ** 2
    )
    metadata = {
        "checkpoint": str(_checkpoint_file(args.checkpoint)),
        "item_name": args.item_name,
        "history_indices_internal": list(sample.capture_rgb_indices),
        "target_indices_internal": target_indices,
        "source_rgb_frame_multiplier": 5,
        "history_source_rgb_indices": [
            index * 5 for index in sample.capture_rgb_indices
        ],
        "target_source_rgb_indices": [index * 5 for index in target_indices],
        "rgb_mse": float(mse.item()),
        "psnr": float((-10.0 * torch.log10(mse.clamp_min(1e-12))).item()),
        "memory_norm": float(memory.float().norm(dim=-1).mean().item()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
