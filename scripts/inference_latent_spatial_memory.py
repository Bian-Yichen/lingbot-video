from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import imageio.v3 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.latent_spatial_memory.controlnet import (  # noqa: E402
    LingBotLatentMemoryControlNet,
)
from lingbot_video.latent_spatial_memory.data import (  # noqa: E402
    LocalRoomTourIndex,
    LongTrajectorySampleConfig,
    SOURCE_FRAME_STRIDE,
    VipeRoomTourItem,
    normalize_preloaded_rgb,
)
from lingbot_video.latent_spatial_memory.geometry import (  # noqa: E402
    make_plucker_rays,
    scale_intrinsics,
)
from lingbot_video.latent_spatial_memory.inference import (  # noqa: E402
    denoise_latent_memory_chunk,
    predict_metric_depth,
)
from lingbot_video.latent_spatial_memory.model import (  # noqa: E402
    LatentMetricDepthHead,
    LingBotVideoLatentMemoryModel,
)
from lingbot_video.latent_spatial_memory.training import (  # noqa: E402
    MemoryTrainingConfig,
    _downsample_depth_and_valid,
    build_capture_memories,
    encode_capture_latents,
    encode_reference_latents,
    encode_video_latents,
)
from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline  # noqa: E402
from lingbot_video.runner import _patch_qwen3vl_from_pretrained  # noqa: E402
from lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModel,
)


logger = logging.getLogger("lingbot_video.inference_latent_spatial_memory")


INHERITED_ARGUMENTS = {
    "model_dir",
    "dataset_root",
    "prompt",
    "height",
    "width",
    "capture_clips",
    "capture_clip_rgb_frames",
    "preceding_rgb_frames",
    "reference_frames",
    "history_min_frames",
    "history_max_frames",
    "latent_frames_per_chunk",
    "min_depth",
    "max_depth",
    "depth_edge_threshold",
    "memory_consistency_threshold",
    "memory_max_points",
    "memory_voxel_size",
    "timestep_shift",
    "control_block_indices",
    "capture_encode_batch_size",
    "mixed_precision",
    "lora_rank",
    "lora_alpha",
}


def _resolve_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_dir():
        checkpoint = checkpoint / "trainable_components.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}; pass either "
            "trainable_components.pt or its containing checkpoint directory"
        )
    return checkpoint.resolve()


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return payload


def _inherited_defaults() -> tuple[dict[str, Any], Path | None]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--checkpoint", default=None)
    bootstrap.add_argument("--config", default=None)
    known, _ = bootstrap.parse_known_args()
    if not known.checkpoint:
        return {}, None
    checkpoint = _resolve_checkpoint(known.checkpoint)
    inherited: dict[str, Any] = {}
    training_args = checkpoint.parent / "training_args.json"
    if training_args.is_file():
        inherited.update(_read_json_object(training_args))
    if known.config:
        inherited.update(_read_json_object(Path(known.config)))
    return {
        key: value
        for key, value in inherited.items()
        if key in INHERITED_ARGUMENTS
    }, checkpoint


def parse_args() -> argparse.Namespace:
    inherited, checkpoint = _inherited_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one training-shaped latent-spatial-memory sample with "
            "full multi-step denoising."
        )
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument(
        "--dataset_root",
        default="/data/bianyichen/H-hdd/AnyReconProDataset_labeled_2",
    )
    parser.add_argument("--item_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default="An indoor room tour.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--capture_clips", type=int, default=16)
    parser.add_argument("--capture_clip_rgb_frames", type=int, default=9)
    parser.add_argument("--preceding_rgb_frames", type=int, default=8)
    parser.add_argument("--reference_frames", type=int, default=4)
    parser.add_argument("--history_min_frames", type=int, default=256)
    parser.add_argument("--history_max_frames", type=int, default=4096)
    parser.add_argument("--latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--depth_edge_threshold", type=float, default=0.08)
    parser.add_argument("--memory_consistency_threshold", type=float, default=0.15)
    parser.add_argument("--memory_max_points", type=int, default=750_000)
    parser.add_argument("--memory_voxel_size", type=float, default=0.0)
    parser.add_argument("--timestep_shift", type=float, default=5.0)
    parser.add_argument("--control_block_indices", default="0,3,6,9,12,15,18,21")
    parser.add_argument("--capture_encode_batch_size", type=int, default=2)
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sample_seed", type=int, default=42)
    parser.add_argument("--generation_seed", type=int, default=42)
    parser.add_argument(
        "--target_start",
        type=int,
        default=None,
        help=(
            "Optional internal target index. Internal index i corresponds to "
            f"source RGB index {SOURCE_FRAME_STRIDE}*i. If omitted, sample it "
            "exactly like training."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output FPS. Defaults to source FPS / 5 when metadata provides it, else 6.",
    )
    parser.set_defaults(**inherited)
    args = parser.parse_args()
    args.checkpoint = checkpoint or _resolve_checkpoint(args.checkpoint)
    args.item_name = args.item_name.strip().rstrip("/")
    if not args.model_dir:
        parser.error(
            "--model_dir is required unless it is available in the checkpoint's "
            "training_args.json or --config"
        )
    if args.height % 16 or args.width % 16:
        parser.error("--height and --width must be multiples of 16")
    if args.steps < 1:
        parser.error("--steps must be positive")
    sample_config = _sample_config(args)
    sample_config.validate()
    return args


def _sample_config(args: argparse.Namespace) -> LongTrajectorySampleConfig:
    return LongTrajectorySampleConfig(
        height=args.height,
        width=args.width,
        capture_clips=args.capture_clips,
        capture_clip_rgb_frames=args.capture_clip_rgb_frames,
        preceding_rgb_frames=args.preceding_rgb_frames,
        reference_frames=args.reference_frames,
        history_min_frames=args.history_min_frames,
        history_max_frames=args.history_max_frames,
        latent_frames_per_chunk=args.latent_frames_per_chunk,
        samples_per_item=1,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )


def _memory_config(args: argparse.Namespace) -> MemoryTrainingConfig:
    # Observed capture RGB-D is clean at inference. Stage 1 training uses these
    # same values; Stage 2's synthetic memory corruption is a train-only
    # robustness augmentation and must not be applied to real capture memory.
    return MemoryTrainingConfig(
        latent_frames_per_chunk=args.latent_frames_per_chunk,
        teacher_memory_probability=1.0,
        memory_update_noise_std=0.0,
        memory_update_dropout_probability=0.0,
        timestep_shift=args.timestep_shift,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        depth_edge_threshold=args.depth_edge_threshold,
        memory_consistency_threshold=args.memory_consistency_threshold,
        memory_max_points=args.memory_max_points,
        memory_voxel_size=args.memory_voxel_size,
        capture_encode_batch_size=args.capture_encode_batch_size,
    )


class _TargetStartRandom:
    """Random wrapper that can force sample()'s first target-window choice.

    Do not subclass ``random.Random`` with an extra constructor argument:
    CPython 3.10's C-level Random constructor rejects more than one argument
    before a subclass ``__init__`` can consume it.
    """

    def __init__(self, seed: int, target_start: int | None) -> None:
        self._random = random.Random(seed)
        self.target_start = target_start
        self._used_forced_choice = False

    def choice(self, sequence: Sequence[int]) -> int:
        if self.target_start is not None and not self._used_forced_choice:
            self._used_forced_choice = True
            if self.target_start not in sequence:
                first = int(sequence[0]) if sequence else None
                last = int(sequence[-1]) if sequence else None
                raise ValueError(
                    f"target_start={self.target_start} is invalid for this item; "
                    f"valid range is {first}..{last}"
                )
            return int(self.target_start)
        return int(self._random.choice(sequence))

    def randint(self, start: int, end: int) -> int:
        return int(self._random.randint(start, end))


def _dtype(name: str) -> torch.dtype:
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _load_model_and_pipeline(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    LingBotVideoLatentMemoryModel,
    LingBotVideoPipeline,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    stage = str(payload.get("stage", "side_branch"))
    raw_indices = payload.get("control_block_indices", args.control_block_indices)
    if isinstance(raw_indices, str):
        block_indices = tuple(
            int(value.strip())
            for value in raw_indices.split(",")
            if value.strip()
        )
    else:
        block_indices = tuple(int(value) for value in raw_indices)
    compute_dtype = _dtype(args.mixed_precision)
    backbone = LingBotVideoTransformer3DModel.from_pretrained(
        args.model_dir,
        subfolder="transformer",
        torch_dtype=compute_dtype,
    )
    latent_channels = int(backbone.config.in_channels)
    controlnet = LingBotLatentMemoryControlNet.from_backbone(
        backbone,
        block_indices,
    )
    if stage == "lora":
        from peft import LoraConfig, get_peft_model

        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=int(payload.get("lora_rank", args.lora_rank)),
                lora_alpha=int(payload.get("lora_alpha", args.lora_alpha)),
                lora_dropout=0.0,
                target_modules=["to_q", "to_k", "to_v", "to_out"],
                bias="none",
            ),
        )
    model = LingBotVideoLatentMemoryModel(
        backbone,
        controlnet,
        LatentMetricDepthHead(latent_channels),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_required = [
        name
        for name in missing
        if name.startswith(("controlnet.", "depth_head."))
        or "lora_" in name
    ]
    if missing_required or unexpected:
        raise RuntimeError(
            "checkpoint does not match the latent-memory model: "
            f"missing_required={missing_required}, unexpected={unexpected}"
        )

    patch_context = (
        _patch_qwen3vl_from_pretrained()
        if _patch_qwen3vl_from_pretrained is not None
        else contextlib.nullcontext()
    )
    with patch_context:
        pipe = LingBotVideoPipeline.from_pretrained(
            args.model_dir,
            transformer=backbone,
            trust_remote_code=True,
            torch_dtype={
                "default": compute_dtype,
                "transformer": compute_dtype,
                "text_encoder": compute_dtype,
                "vae": torch.float32,
            },
        )
    pipe.text_encoder.to(device)
    with torch.no_grad():
        prompt_embeds, prompt_mask = pipe.encode_prompt(args.prompt, device=device)
    pipe.text_encoder.to("cpu")
    pipe.vae.requires_grad_(False)
    pipe.vae.eval().to(device)
    model.requires_grad_(False)
    model.eval().to(device)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    checkpoint_info = {
        "path": str(args.checkpoint),
        "stage": stage,
        "step": int(payload.get("step", -1)),
        "control_block_indices": list(block_indices),
    }
    return model, pipe, prompt_embeds.detach(), prompt_mask.detach(), checkpoint_info


def _batch_sample(
    sample: dict[str, torch.Tensor | str],
    device: torch.device,
) -> dict[str, torch.Tensor | str]:
    batch = {
        key: value.unsqueeze(0).to(device, non_blocking=True)
        if torch.is_tensor(value)
        else value
        for key, value in sample.items()
    }
    normalize_preloaded_rgb(batch)
    return batch


def _encode_training_inputs(
    vae: torch.nn.Module,
    batch: dict[str, torch.Tensor | str],
    config: MemoryTrainingConfig,
    *,
    generator: torch.Generator,
    mixed_precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = batch["target_rgb"].device
    autocast_enabled = (
        device.type == "cuda"
        and mixed_precision in {"fp16", "bf16"}
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=_dtype(mixed_precision),
        enabled=autocast_enabled,
    ):
        capture_latents = encode_capture_latents(
            vae,
            batch["capture_rgb"],
            micro_batch_size=config.capture_encode_batch_size,
            generator=generator,
        )
        preceding_latents = encode_video_latents(
            vae,
            batch["preceding_rgb"],
            generator=generator,
        )
        reference_latents = encode_reference_latents(
            vae,
            batch["reference_rgb"],
            micro_batch_size=config.capture_encode_batch_size,
            generator=generator,
        )
        target_latents = encode_video_latents(
            vae,
            batch["target_rgb"],
            generator=generator,
        )
    if target_latents.shape[2] != config.latent_frames_per_chunk:
        raise ValueError(
            f"expected {config.latent_frames_per_chunk} target latents, "
            f"got {target_latents.shape[2]}"
        )
    if capture_latents.shape[1] != batch["capture_c2w"].shape[1]:
        raise ValueError("capture VAE anchors and capture geometry do not align")
    if preceding_latents.shape[2] != batch["preceding_c2w"].shape[1]:
        raise ValueError("preceding VAE anchors and geometry do not align")
    target_latents = target_latents.clone()
    target_latents[:, :, 0] = capture_latents[:, -1]
    return (
        capture_latents,
        preceding_latents,
        reference_latents,
        target_latents,
    )


def _read_training_conditions(
    memories,
    batch: dict[str, torch.Tensor | str],
    latent_hw: tuple[int, int],
    image_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_k = scale_intrinsics(
        batch["target_intrinsics"],
        image_hw,
        latent_hw,
    )
    preceding_k = scale_intrinsics(
        batch["preceding_intrinsics"],
        image_hw,
        latent_hw,
    )
    target_rays = make_plucker_rays(
        batch["target_c2w"],
        target_k,
        *latent_hw,
    ).transpose(1, 2)
    preceding_rays = make_plucker_rays(
        batch["preceding_c2w"],
        preceding_k,
        *latent_hw,
    ).transpose(1, 2)
    condition_c2w = torch.cat(
        (batch["target_c2w"], batch["preceding_c2w"]),
        dim=1,
    )
    condition_k = torch.cat((target_k, preceding_k), dim=1)
    readouts = [
        memory.read(condition_c2w[index], condition_k[index], latent_hw)
        for index, memory in enumerate(memories)
    ]
    return (
        torch.stack([readout.features for readout in readouts]),
        torch.stack([readout.visibility for readout in readouts]),
        torch.stack([readout.depth for readout in readouts]),
        target_rays,
        preceding_rays,
    )


def _video_uint8(frames: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(frames * 255.0), 0, 255).astype(np.uint8)


def _write_video(path: Path, frames: np.ndarray, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        path,
        _video_uint8(frames),
        fps=float(fps),
        codec="libx264",
        pixelformat="yuv420p",
    )


def _psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.square(prediction.astype(np.float64) - target).mean())
    if mse == 0:
        return float("inf")
    return float(-10.0 * math.log10(mse))


def _depth_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, float]:
    mask = (
        valid.bool()
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & (prediction > 0)
        & (target > 0)
    )
    if not bool(mask.any()):
        return {"depth_abs_rel": float("nan"), "depth_rmse": float("nan")}
    difference = prediction[mask].float() - target[mask].float()
    return {
        "depth_abs_rel": float(
            (difference.abs() / target[mask].float().clamp_min(1e-6)).mean().item()
        ),
        "depth_rmse": float(difference.square().mean().sqrt().item()),
    }


def _infer_fps(metadata: dict[str, Any], requested: float | None) -> float:
    if requested is not None:
        return float(requested)
    candidates = (
        metadata.get("fps"),
        metadata.get("source_fps"),
        metadata.get("video_fps"),
    )
    for value in candidates:
        if isinstance(value, (int, float)) and value > 0:
            return float(value) / float(SOURCE_FRAME_STRIDE)
    return 6.0


def _indices(value: torch.Tensor) -> list[int]:
    return [int(index) for index in value.reshape(-1).cpu().tolist()]


def _source_indices(
    item: VipeRoomTourItem,
    internal_indices: Sequence[int],
) -> list[int]:
    return [
        int(
            item.rgb_source_index_by_index.get(
                index,
                index * SOURCE_FRAME_STRIDE,
            )
        )
        for index in internal_indices
    ]


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("latent-spatial-memory inference requires a CUDA GPU")
    device = torch.device("cuda")
    torch.manual_seed(args.generation_seed)
    generator = torch.Generator(device=device).manual_seed(args.generation_seed)
    sample_config = _sample_config(args)
    memory_config = _memory_config(args)

    item_index = LocalRoomTourIndex(args.dataset_root)
    logger.info("loading local scene %s", args.item_name)
    item = VipeRoomTourItem(item_index.item_path(args.item_name))
    sample_rng = _TargetStartRandom(args.sample_seed, args.target_start)
    sample = item.sample(sample_config, sample_rng)
    target_start = int(sample["target_rgb_indices"][0].item())
    logger.info(
        "sampled target_start=%d (source RGB=%d), capture_clips=%d, target_rgb=%d",
        target_start,
        item.rgb_source_index_by_index[target_start],
        args.capture_clips,
        sample_config.target_rgb_frames,
    )

    model, pipe, prompt_embeds, prompt_mask, checkpoint_info = (
        _load_model_and_pipeline(args, device)
    )
    batch = _batch_sample(sample, device)
    (
        capture_latents,
        preceding_latents,
        reference_latents,
        target_latents,
    ) = _encode_training_inputs(
        pipe.vae,
        batch,
        memory_config,
        generator=generator,
        mixed_precision=args.mixed_precision,
    )
    image_hw = tuple(int(value) for value in batch["image_hw"][0].tolist())
    latent_hw = (target_latents.shape[-2], target_latents.shape[-1])
    memories = build_capture_memories(
        capture_latents,
        batch,
        image_hw=image_hw,
        config=memory_config,
    )
    (
        memory_latents,
        memory_visibility,
        projected_depth,
        target_rays,
        preceding_rays,
    ) = _read_training_conditions(
        memories,
        batch,
        latent_hw,
        image_hw,
    )
    target_frames = target_latents.shape[2]
    target_depth_latent, target_valid_latent = _downsample_depth_and_valid(
        batch["target_depth"],
        batch["target_valid"],
        latent_hw,
    )
    logger.info(
        "starting %d-step denoising: target_latents=%s memory_points=%d "
        "visible_fraction=%.4f",
        args.steps,
        tuple(target_latents.shape),
        len(memories[0]),
        float(memory_visibility[:, :, :target_frames].mean().item()),
    )
    generated_latents = denoise_latent_memory_chunk(
        model,
        pipe.scheduler,
        clean_prefix=target_latents[:, :, :1],
        memory_latents=memory_latents,
        memory_visibility=memory_visibility,
        target_rays=target_rays,
        preceding_latents=preceding_latents,
        preceding_rays=preceding_rays,
        reference_latents=reference_latents,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        num_inference_steps=args.steps,
        timestep_shift=args.timestep_shift,
        generator=generator,
    )
    predicted_depth = predict_metric_depth(
        model,
        generated_latents,
        target_rays,
        projected_depth[:, :, :target_frames],
        memory_visibility[:, :, :target_frames],
    )

    logger.info("decoding generated and teacher target latents")
    with torch.no_grad():
        generated_rgb = pipe._decode_latents(generated_latents)[0]
        vae_target_rgb = pipe._decode_latents(target_latents)[0]
    ground_truth_rgb = (
        batch["target_rgb"][0]
        .permute(1, 2, 3, 0)
        .float()
        .clamp(0, 1)
        .cpu()
        .numpy()
    )
    generated_rgb = np.asarray(generated_rgb, dtype=np.float32)
    vae_target_rgb = np.asarray(vae_target_rgb, dtype=np.float32)
    if generated_rgb.shape != ground_truth_rgb.shape:
        raise RuntimeError(
            f"decoded/GT RGB shapes differ: {generated_rgb.shape} vs "
            f"{ground_truth_rgb.shape}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = _infer_fps(item.metadata, args.fps)
    _write_video(output_dir / "generated.mp4", generated_rgb, fps)
    _write_video(output_dir / "ground_truth.mp4", ground_truth_rgb, fps)
    _write_video(output_dir / "vae_reconstruction.mp4", vae_target_rgb, fps)
    capture_rgb = (
        batch["capture_rgb"][0]
        .permute(0, 2, 3, 4, 1)
        .reshape(-1, args.height, args.width, 3)
        .float()
        .cpu()
        .numpy()
    )
    preceding_rgb = (
        batch["preceding_rgb"][0]
        .permute(1, 2, 3, 0)
        .float()
        .cpu()
        .numpy()
    )
    reference_rgb = (
        batch["reference_rgb"][0]
        .permute(0, 2, 3, 1)
        .float()
        .cpu()
        .numpy()
    )
    _write_video(output_dir / "conditioning_capture_clips.mp4", capture_rgb, fps)
    _write_video(output_dir / "conditioning_preceding.mp4", preceding_rgb, fps)
    if reference_rgb.shape[0]:
        iio.imwrite(
            output_dir / "conditioning_references.png",
            _video_uint8(np.concatenate(list(reference_rgb), axis=1)),
        )
    comparison = np.concatenate(
        (ground_truth_rgb, vae_target_rgb, generated_rgb),
        axis=2,
    )
    _write_video(output_dir / "comparison_gt_vae_generated.mp4", comparison, fps)

    target_internal = _indices(batch["target_rgb_indices"][0])
    target_latent_internal = _indices(batch["target_latent_indices"][0])
    capture_internal = _indices(batch["capture_indices"][0])
    capture_starts = _indices(batch["capture_clip_starts"][0])
    reference_internal = _indices(batch["reference_indices"][0])
    preceding_internal = _indices(batch["preceding_rgb_indices"][0])
    metrics = {
        "rgb_psnr_generated_vs_gt": _psnr(generated_rgb, ground_truth_rgb),
        "rgb_psnr_vae_vs_gt": _psnr(vae_target_rgb, ground_truth_rgb),
        "latent_mse_excluding_overlap": float(
            torch.nn.functional.mse_loss(
                generated_latents[:, :, 1:].float(),
                target_latents[:, :, 1:].float(),
            ).item()
        ),
        "memory_points": len(memories[0]),
        "visible_fraction": float(
            memory_visibility[:, :, :target_frames].mean().item()
        ),
        **_depth_metrics(
            predicted_depth,
            target_depth_latent,
            target_valid_latent,
        ),
    }
    metadata = {
        "scene": args.item_name,
        "dataset_root": args.dataset_root,
        "checkpoint": checkpoint_info,
        "model_dir": args.model_dir,
        "prompt": args.prompt,
        "sample_seed": args.sample_seed,
        "generation_seed": args.generation_seed,
        "steps": args.steps,
        "timestep_shift": args.timestep_shift,
        "fps": fps,
        "image_hw": list(image_hw),
        "latent_hw": list(latent_hw),
        "capture_clip_starts_internal": capture_starts,
        "capture_clip_starts_source": _source_indices(item, capture_starts),
        "capture_anchor_indices_internal": capture_internal,
        "capture_anchor_indices_source": _source_indices(item, capture_internal),
        "reference_indices_internal": reference_internal,
        "reference_indices_source": _source_indices(item, reference_internal),
        "preceding_rgb_indices_internal": preceding_internal,
        "preceding_rgb_indices_source": _source_indices(item, preceding_internal),
        "target_rgb_indices_internal": target_internal,
        "target_rgb_indices_source": _source_indices(item, target_internal),
        "target_latent_indices_internal": target_latent_internal,
        "target_latent_indices_source": _source_indices(
            item,
            target_latent_internal,
        ),
        "metrics": metrics,
        "comparison_order": ["ground_truth", "vae_reconstruction", "generated"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "geometry_and_depth.npz",
        predicted_depth=predicted_depth.cpu().numpy(),
        target_depth_latent=target_depth_latent.float().cpu().numpy(),
        target_valid_latent=target_valid_latent.cpu().numpy(),
        target_depth=batch["target_depth"].float().cpu().numpy(),
        target_valid=batch["target_valid"].cpu().numpy(),
        target_c2w=batch["target_c2w"].float().cpu().numpy(),
        target_intrinsics=batch["target_intrinsics"].float().cpu().numpy(),
        projected_memory_depth=projected_depth[
            :, :, :target_frames
        ].float().cpu().numpy(),
        memory_visibility=memory_visibility[
            :, :, :target_frames
        ].float().cpu().numpy(),
    )
    logger.info("inference complete: %s", output_dir)
    logger.info("metrics: %s", json.dumps(metrics, allow_nan=True))


if __name__ == "__main__":
    main()
