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
from lingbot_video.geometry_aware_memory.three_drae import (  # noqa: E402
    ThreeDRAEConfig,
    ThreeDRAEViewCurriculum,
    WanThreeDRAEModel,
)
from lingbot_video.geometry_aware_memory.wan_latent_reconstruction import (  # noqa: E402,E501
    WanDecoderBridge,
)
from lingbot_video.geometry_aware_memory.wan_latent_training import (  # noqa: E402
    prepare_wan_latent_batch,
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
        description=(
            "Query a trained Wan-3DRAE scene memory at held-out cameras and "
            "decode the predicted Wan latents into RGB views."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--item_name", required=True)
    parser.add_argument("--output_dir", default="outputs/memory_3drae_eval")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--query_blocks", type=int, default=1)
    parser.add_argument("--query_views", type=int, default=None)
    parser.add_argument("--query_views_start", type=int, default=2)
    parser.add_argument("--query_views_end", type=int, default=8)
    parser.add_argument("--local_window_rgb_frames", type=int, default=81)
    parser.add_argument("--memory_views_min", type=int, default=None)
    parser.add_argument("--memory_views_max", type=int, default=None)
    parser.add_argument("--history_views_start_min", type=int, default=2)
    parser.add_argument("--history_views_start_max", type=int, default=4)
    parser.add_argument("--history_views_end_min", type=int, default=12)
    parser.add_argument("--history_views_end_max", type=int, default=32)
    parser.add_argument("--view_curriculum_epochs", type=int, default=8)
    parser.add_argument("--retrieval_rotation_weight", type=float, default=0.25)
    parser.add_argument("--retrieval_temperature", type=float, default=0.25)
    parser.add_argument("--vae_encode_chunk_rgb_frames", type=int, default=16)
    parser.add_argument("--local_window_start", type=int, default=None)
    parser.add_argument("--target_start", type=int, default=None)
    parser.add_argument("--memory_view_count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage",
        choices=["auto", "1", "2"],
        default="auto",
    )
    parser.add_argument(
        "--sampling_profile",
        choices=["checkpoint", "final"],
        default="checkpoint",
        help=(
            "Use the view counts reached by this checkpoint, or the final "
            "curriculum counts. Explicit --query_views/--memory_views_* "
            "always override the profile."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    if not args.model_dir or not args.dataset_root:
        parser.error("model_dir and dataset_root are required")
    if (args.memory_views_min is None) != (args.memory_views_max is None):
        parser.error("memory_views_min and memory_views_max must be set together")
    if args.query_views is not None and args.query_views < 1:
        parser.error("query_views must be positive")
    if args.memory_views_min is not None and (
        args.memory_views_min < 1
        or args.memory_views_max < args.memory_views_min
    ):
        parser.error("memory view range must be positive and ordered")
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
        path = path / "memory_3drae.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _normalize_model_config(payload: dict[str, Any]) -> dict[str, Any]:
    output = dict(payload)
    for name in (
        "latent_patch_size",
        "vae_latents_mean",
        "vae_latents_std",
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


def _select_stage(requested: str, checkpoint_payload: dict[str, Any]) -> int:
    if requested != "auto":
        return int(requested)
    if "training_stage" in checkpoint_payload:
        return int(checkpoint_payload["training_stage"])
    training = checkpoint_payload.get("training_config", {})
    stage1_epochs = int(training.get("stage1_epochs", 0))
    next_epoch = int(checkpoint_payload.get("next_epoch", 0))
    return 1 if next_epoch < stage1_epochs else 2


def _resolve_sampling_counts(
    args: argparse.Namespace,
    checkpoint_payload: dict[str, Any],
) -> tuple[int, int, int, int]:
    training = checkpoint_payload.get("training_config", {})

    def value(name: str) -> int:
        return int(training.get(name, getattr(args, name)))

    curriculum = ThreeDRAEViewCurriculum(
        history_start_min=value("history_views_start_min"),
        history_start_max=value("history_views_start_max"),
        history_end_min=value("history_views_end_min"),
        history_end_max=value("history_views_end_max"),
        query_start=value("query_views_start"),
        query_end=value("query_views_end"),
        curriculum_epochs=value("view_curriculum_epochs"),
    )
    curriculum.validate()
    if args.sampling_profile == "checkpoint":
        curriculum_epoch = int(checkpoint_payload.get("next_epoch", 0))
        if (
            "next_iteration_in_epoch" in checkpoint_payload
            and int(checkpoint_payload["next_iteration_in_epoch"]) == 0
            and curriculum_epoch > 0
        ):
            # An epoch-boundary checkpoint has finished the preceding epoch;
            # test the view counts that its weights have actually seen.
            curriculum_epoch -= 1
    else:
        curriculum_epoch = curriculum.curriculum_epochs - 1
    memory_min, memory_max, query_views = curriculum.values_for_epoch(
        curriculum_epoch
    )
    if args.memory_views_min is not None:
        memory_min = int(args.memory_views_min)
        memory_max = int(args.memory_views_max)
    if args.query_views is not None:
        query_views = int(args.query_views)
    return memory_min, memory_max, query_views, curriculum_epoch


def _load_model_and_encoder(
    args: argparse.Namespace,
    checkpoint_payload: dict[str, Any],
) -> tuple[WanThreeDRAEModel, torch.nn.Module]:
    model_config = ThreeDRAEConfig(
        **_normalize_model_config(checkpoint_payload["model_config"])
    )
    model_config.validate()
    if (args.height, args.width) != (
        model_config.image_height,
        model_config.image_width,
    ):
        raise ValueError("inference resolution does not match checkpoint")
    vae = AutoencoderKLWan.from_pretrained(
        args.model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval().requires_grad_(False)
    if vae.config.patch_size is not None:
        raise NotImplementedError("patchified Wan VAE is not supported")
    if int(vae.config.scale_factor_spatial) != model_config.vae_spatial_stride:
        raise ValueError("checkpoint spatial stride does not match Wan VAE")
    pretrained_means = tuple(float(value) for value in vae.config.latents_mean)
    pretrained_stds = tuple(float(value) for value in vae.config.latents_std)
    if (
        pretrained_means != model_config.vae_latents_mean
        or pretrained_stds != model_config.vae_latents_std
    ):
        raise ValueError("checkpoint latent normalization does not match Wan VAE")
    bridge = WanDecoderBridge(
        vae.post_quant_conv,
        vae.decoder,
        latent_channels=model_config.latent_channels,
        refiner_hidden_size=model_config.decoder_refiner_hidden_size,
        lora_rank=model_config.decoder_lora_rank,
        lora_alpha=model_config.decoder_lora_alpha,
    )
    vae.post_quant_conv = None
    vae.decoder = None
    model = WanThreeDRAEModel(bridge, model_config)
    model.load_experiment_state_dict(checkpoint_payload["model"])
    return model, vae


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Wan 3DRAE inference requires CUDA")
    device = torch.device("cuda")
    compute_dtype = _dtype(args.mixed_precision)
    checkpoint_path = _checkpoint_file(args.checkpoint)
    checkpoint_payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    memory_views_min, memory_views_max, query_views, curriculum_epoch = (
        _resolve_sampling_counts(args, checkpoint_payload)
    )
    model, vae_encoder = _load_model_and_encoder(args, checkpoint_payload)
    prediction_stage = _select_stage(args.stage, checkpoint_payload)
    model.set_training_stage(prediction_stage)
    model.eval().requires_grad_(False).to(device)
    vae_encoder.to(device)

    sample_config = GeometryMemorySampleConfig(
        height=args.height,
        width=args.width,
        target_rgb_frames=query_views,
        query_blocks=args.query_blocks,
        local_window_rgb_frames=args.local_window_rgb_frames,
        memory_views_min=memory_views_min,
        memory_views_max=memory_views_max,
        retrieval_rotation_weight=args.retrieval_rotation_weight,
        retrieval_temperature=args.retrieval_temperature,
    )
    sample_config.validate()
    item = VipeRoomTourItem(Path(args.dataset_root) / args.item_name)
    sample = item.make_sample(
        sample_config,
        random.Random(args.seed),
        local_window_start=args.local_window_start,
        target_start=args.target_start,
        memory_view_count=args.memory_view_count,
    )
    sample = item.preload_sample(sample)
    prepared = prepare_wan_latent_batch(
        sample,
        vae=vae_encoder,
        vae_encode_chunk_rgb_frames=args.vae_encode_chunk_rgb_frames,
        device=device,
        compute_dtype=compute_dtype,
    )

    with torch.no_grad(), torch.autocast(
        "cuda",
        dtype=compute_dtype,
        enabled=compute_dtype != torch.float32,
    ):
        predicted_latents, prediction, memory, visibility = model(
            prepared.history_latents.to(device=device, dtype=compute_dtype),
            prepared.history_c2w.to(device),
            prepared.history_intrinsics.to(device),
            prepared.target_c2w.to(device),
            prepared.target_intrinsics.to(device),
        )
        target_latents = prepared.target_latents.to(
            device=device,
            dtype=compute_dtype,
        ).permute(0, 2, 1, 3, 4)
        model.set_training_stage(1)
        vae_reconstruction = model.wan_decoder(
            model.normalized_to_native(target_latents)
        ).add(1.0).mul(0.5)
        model.set_training_stage(prediction_stage)

    output_dir = Path(args.output_dir) / args.item_name
    output_dir.mkdir(parents=True, exist_ok=True)
    target_indices = [
        index for block in sample.query_rgb_blocks for index in block
    ]
    for view_index, frame_index in enumerate(target_indices):
        prefix = output_dir / f"frame-{frame_index:06d}"
        _save_rgb(
            prefix.with_name(prefix.name + "-prediction.png"),
            prediction[0, view_index],
        )
        _save_rgb(
            prefix.with_name(prefix.name + "-vae-reconstruction.png"),
            vae_reconstruction[0, view_index],
        )
        _save_rgb(
            prefix.with_name(prefix.name + "-target.png"),
            prepared.target_rgb[0, view_index],
        )
    torch.save(
        {
            "predicted_standardized_latents": predicted_latents.float().cpu(),
            "target_standardized_latents": target_latents.float().cpu(),
            "memory_tokens": memory.float().cpu(),
        },
        output_dir / "latents_and_memory.pt",
    )

    target_rgb = prepared.target_rgb.float()
    prediction_cpu = prediction.detach().float().cpu()
    vae_reconstruction_cpu = vae_reconstruction.detach().float().cpu()
    rgb_mse = torch.mean((prediction_cpu - target_rgb) ** 2)
    vae_rgb_mse = torch.mean((vae_reconstruction_cpu - target_rgb) ** 2)
    latent_mse = torch.mean(
        (
            predicted_latents.detach().float().cpu()
            - target_latents.detach().float().cpu()
        )
        ** 2
    )
    target_start = min(target_indices)
    target_end = max(target_indices)
    before = [index for index in sample.capture_rgb_indices if index < target_start]
    after = [index for index in sample.capture_rgb_indices if index > target_end]
    metadata = {
        "checkpoint": str(checkpoint_path),
        "prediction_stage": prediction_stage,
        "sampling_profile": args.sampling_profile,
        "sampling_curriculum_epoch_zero_based": curriculum_epoch,
        "requested_history_view_range": [memory_views_min, memory_views_max],
        "requested_query_views": query_views,
        "item_name": args.item_name,
        "history_indices_internal": list(sample.capture_rgb_indices),
        "target_indices_internal": target_indices,
        "history_views_before_target": len(before),
        "history_views_after_target": len(after),
        "history_target_overlap": len(
            set(sample.capture_rgb_indices).intersection(target_indices)
        ),
        "source_rgb_frame_multiplier": 5,
        "history_source_rgb_indices": [
            index * 5 for index in sample.capture_rgb_indices
        ],
        "target_source_rgb_indices": [index * 5 for index in target_indices],
        "latent_mse": float(latent_mse.item()),
        "rgb_mse": float(rgb_mse.item()),
        "psnr": float((-10.0 * torch.log10(rgb_mse.clamp_min(1e-12))).item()),
        "vae_reconstruction_rgb_mse": float(vae_rgb_mse.item()),
        "vae_reconstruction_psnr": float(
            (-10.0 * torch.log10(vae_rgb_mse.clamp_min(1e-12))).item()
        ),
        "memory_shape": list(memory.shape),
        "memory_norm": float(memory.float().norm(dim=-1).mean().item()),
        "visible_history_fraction": float(visibility.float().mean().item()),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
