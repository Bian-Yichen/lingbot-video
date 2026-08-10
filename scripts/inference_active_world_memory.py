from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.active_world_memory.data import (  # noqa: E402
    ActiveMemorySampleConfig,
    VipeRoomTourItem,
)
from lingbot_video.active_world_memory.agent import QUERY_TYPES  # noqa: E402
from lingbot_video.active_world_memory.geometry import (  # noqa: E402
    intrinsics_vector_to_matrix,
    normalize_c2w_to_first_capture,
    resize_crop_intrinsics,
)
from lingbot_video.active_world_memory.model import (  # noqa: E402
    ActiveWorldMemoryConfig,
    ActiveWorldMemoryModel,
)
from lingbot_video.active_world_memory.lora import (  # noqa: E402
    inject_lora,
    load_lora_state_dict,
)
from lingbot_video.active_world_memory.vae import (  # noqa: E402
    decode_video_latents,
    encode_independent_views,
)
from lingbot_video.pipeline_lingbot_video import (  # noqa: E402
    DEFAULT_NEGATIVE_PROMPT,
    LingBotVideoPipeline,
)
from lingbot_video.runner import _patch_qwen3vl_from_pretrained  # noqa: E402
from lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModel,
)


logger = logging.getLogger("lingbot_video.inference_active_world_memory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--item_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_epoch", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--stop_threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--target_camera_npz",
        default=None,
        help=(
            "Optional novel trajectory with c2w/poses, intrinsics, and optional "
            "times arrays. Without it, a held-out dataset target is generated."
        ),
    )
    parser.add_argument(
        "--target_poses_normalized",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Declare that target_camera_npz c2w matrices are already capture-relative.",
    )
    parser.add_argument(
        "--full_capture_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Index every non-target sparse capture view instead of the training subset.",
    )
    parser.add_argument(
        "--max_capture_views",
        type=int,
        default=0,
        help="Optional stratified cap for full capture; 0 keeps every view.",
    )
    return parser.parse_args()


def _checkpoint_file(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "trainable_components.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _dataclass_from_section(cls, section: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**section)


def _write_video(path: Path, video: torch.Tensor | np.ndarray, fps: int) -> None:
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().numpy()
    if video.dtype != np.uint8:
        video = np.clip(video * 255.0, 0, 255).astype(np.uint8)
    iio.imwrite(path, video, fps=fps)


def _external_target_blocks(
    path: str | Path,
    item: VipeRoomTourItem,
    config: ActiveMemorySampleConfig,
    *,
    origin_index: int,
    already_normalized: bool,
) -> tuple[
    list[tuple[int, ...]],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[tuple[int, int]],
]:
    """Load an arbitrary target trajectory and window it with one-frame overlap."""

    with np.load(path, allow_pickle=False) as payload:
        pose_key = "c2w" if "c2w" in payload else "poses" if "poses" in payload else None
        if pose_key is None or "intrinsics" not in payload:
            raise KeyError("target camera NPZ needs c2w (or poses) and intrinsics")
        c2w = torch.from_numpy(np.asarray(payload[pose_key])).float()
        intrinsics = torch.from_numpy(np.asarray(payload["intrinsics"])).float()
        times = (
            torch.from_numpy(np.asarray(payload["times"])).float()
            if "times" in payload
            else None
        )

    if c2w.ndim == 2:
        c2w = c2w.unsqueeze(0)
    if c2w.ndim != 3 or c2w.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError(f"target c2w must be (T,3,4) or (T,4,4), got {tuple(c2w.shape)}")
    frames = c2w.shape[0]
    if frames < 1:
        raise ValueError("target trajectory is empty")
    if not already_normalized:
        origin = torch.from_numpy(item.pose[origin_index]).float()
        c2w = normalize_c2w_to_first_capture(c2w, origin)

    if intrinsics.shape[-2:] == (3, 3):
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
    else:
        intrinsics = intrinsics_vector_to_matrix(intrinsics)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
    if intrinsics.shape[0] == 1:
        intrinsics = intrinsics.expand(frames, -1, -1).clone()
    if intrinsics.shape != (frames, 3, 3):
        raise ValueError(
            f"target intrinsics must broadcast to (T,3,3), got {tuple(intrinsics.shape)}"
        )
    intrinsics = resize_crop_intrinsics(
        intrinsics, item.source_hw, (config.height, config.width)
    )
    if times is None:
        # By default the novel rollout follows the capture in scene time.
        times = torch.arange(frames, dtype=torch.float32) + float(
            item.indices[-1] - origin_index + 1
        )
    times = times.reshape(-1)
    if times.numel() != frames:
        raise ValueError("target times must contain one value per target pose")

    block_frames = config.target_rgb_frames
    stride = block_frames - 1
    index_blocks: list[tuple[int, ...]] = []
    c2w_blocks: list[torch.Tensor] = []
    intrinsics_blocks: list[torch.Tensor] = []
    time_blocks: list[torch.Tensor] = []
    output_slices: list[tuple[int, int]] = []
    start = 0
    block_index = 0
    while start < frames:
        stop = min(start + block_frames, frames)
        real_indices = list(range(start, stop))
        padded_indices = real_indices + [real_indices[-1]] * (
            block_frames - len(real_indices)
        )
        gather = torch.tensor(padded_indices, dtype=torch.long)
        index_blocks.append(tuple(real_indices))
        c2w_blocks.append(c2w.index_select(0, gather))
        intrinsics_blocks.append(intrinsics.index_select(0, gather))
        time_blocks.append(times.index_select(0, gather))
        drop = 0 if block_index == 0 else 1
        output_slices.append((drop, max(0, len(real_indices) - drop)))
        if stop == frames:
            break
        start += stride
        block_index += 1
    return index_blocks, c2w_blocks, intrinsics_blocks, time_blocks, output_slices


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    checkpoint_path = _checkpoint_file(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    else:
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError("checkpoint has no embedded config; pass --config")
    os.environ["LINGBOT_FUSED_QKV_LINEAR"] = (
        "1" if bool(config.get("fused_qkv_linear", False)) else "0"
    )
    data_config = _dataclass_from_section(ActiveMemorySampleConfig, config.get("data", {}))
    model_config = _dataclass_from_section(
        ActiveWorldMemoryConfig,
        payload.get("active_world_memory_config", config.get("model", {})),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = config.get("optimization", {}).get("mixed_precision", "bf16")
    compute_dtype = {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]

    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        config["model_dir"], subfolder="transformer", torch_dtype=compute_dtype
    )
    attention_environment = "LINGBOT_QWEN_ATTN_IMPLEMENTATION"
    previous_attention = os.environ.get(attention_environment)
    os.environ[attention_environment] = str(
        config.get("qwen_attn_implementation", "sdpa")
    )
    try:
        patch_context = (
            _patch_qwen3vl_from_pretrained()
            if _patch_qwen3vl_from_pretrained is not None
            else contextlib.nullcontext()
        )
        with patch_context:
            pipe = LingBotVideoPipeline.from_pretrained(
                config["model_dir"],
                transformer=transformer,
                trust_remote_code=True,
                local_files_only=bool(config.get("local_files_only", True)),
                torch_dtype={
                    "default": compute_dtype,
                    "transformer": compute_dtype,
                    "text_encoder": compute_dtype,
                    "vae": torch.float32,
                },
            )
    finally:
        if previous_attention is None:
            os.environ.pop(attention_environment, None)
        else:
            os.environ[attention_environment] = previous_attention
    pipe.to(device)
    with torch.no_grad():
        prompt_embeds, prompt_mask = pipe.encode_prompt(
            config.get("prompt", "An indoor room tour."), device=device
        )
        negative_embeds, negative_mask = pipe.encode_prompt(
            config.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT), device=device
        )
    pipe.text_encoder.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = ActiveWorldMemoryModel(pipe.transformer, model_config)
    optimization = config.get("optimization", {})
    backbone_mode = str(optimization.get("backbone_train_mode", "lora"))
    if backbone_mode == "lora":
        if os.environ.get("LINGBOT_FUSED_QKV_LINEAR") == "1":
            raise ValueError(
                "native LoRA requires LINGBOT_FUSED_QKV_LINEAR=0 because the "
                "fused path bypasses wrapped Q/K/V Linear forwards"
            )
        inject_lora(
            model.backbone,
            rank=int(optimization.get("lora_rank", 16)),
            alpha=float(optimization.get("lora_alpha", 16.0)),
            dropout=float(optimization.get("lora_dropout", 0.0)),
            target_suffixes=optimization.get(
                "lora_targets",
                ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"],
            ),
        )
    elif backbone_mode not in {"frozen", "full"}:
        raise ValueError("backbone_train_mode must be 'frozen', 'lora', or 'full'")
    model.to(device).eval()
    missing, unexpected = model.load_state_dict(
        payload["active_world_memory"], strict=False
    )
    active_missing = [name for name in missing if not name.startswith("backbone.")]
    if active_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={active_missing[:20]}, unexpected={unexpected[:20]}"
        )
    if "backbone" in payload:
        model.backbone.load_state_dict(payload["backbone"], strict=False)
    if "backbone_lora" in payload:
        load_lora_state_dict(model.backbone, payload["backbone_lora"])
    elif backbone_mode == "lora":
        raise RuntimeError("checkpoint config requests LoRA but checkpoint has no backbone_lora")

    import random

    item = VipeRoomTourItem(Path(config["dataset_root"]) / args.item_name)
    external_target = args.target_camera_npz is not None
    sample = None
    if external_target:
        if not args.full_capture_memory:
            raise ValueError("external target trajectories require --full_capture_memory")
        image_hw = (data_config.height, data_config.width)
        origin_index = item.indices[0]
        (
            target_index_blocks,
            target_c2w_blocks,
            target_k_blocks,
            target_time_blocks,
            target_output_slices,
        ) = _external_target_blocks(
            args.target_camera_npz,
            item,
            data_config,
            origin_index=origin_index,
            already_normalized=args.target_poses_normalized,
        )
        target_rgb_blocks = None
        capture_span = (item.indices[0], item.indices[-1])
    else:
        sample = item.sample(
            data_config,
            random.Random(args.seed),
            args.sample_epoch,
            load_candidate_rgb=not args.full_capture_memory,
        )
        image_hw = sample.image_hw
        origin_index = sample.candidate_indices[0]
        target_index_blocks = list(sample.target_index_blocks)
        target_c2w_blocks = list(sample.target_c2w_blocks)
        target_k_blocks = list(sample.target_intrinsics_blocks)
        target_time_blocks = list(sample.target_times_blocks)
        target_output_slices = [
            (0, data_config.target_rgb_frames) for _ in target_index_blocks
        ]
        target_rgb_blocks = list(sample.target_rgb_uint8_blocks)
        capture_span = sample.capture_span

    if args.full_capture_memory:
        target_set = (
            set()
            if external_target
            else {index for block in target_index_blocks for index in block}
        )
        candidate_indices = [
            index
            for index in item.indices
            if all(
                abs(index - target) > data_config.target_exclusion_radius
                for target in target_set
            )
        ]
        if args.max_capture_views > 0 and len(candidate_indices) > args.max_capture_views:
            candidate_indices = VipeRoomTourItem._stratified_sample(
                candidate_indices,
                args.max_capture_views,
                random.Random(args.seed + 17),
            )
        full_c2w, full_k = item.cameras(
            candidate_indices, image_hw, origin_index=origin_index
        )
        # Only thumbnails are resident for the long bank. Full-resolution RGB
        # is decoded lazily after the agent has selected 2..24 views.
        candidate_rgb = item.read_video_uint8(
            candidate_indices, (128, 224)
        ).permute(1, 0, 2, 3).unsqueeze(0)
        candidate_c2w = full_c2w.unsqueeze(0).to(device)
        candidate_k = full_k.unsqueeze(0).to(device)
        candidate_times = torch.tensor(
            [index - origin_index for index in candidate_indices],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        episode_ids = VipeRoomTourItem._episode_ids(
            candidate_indices, data_config.episode_size
        ).unsqueeze(0).to(device)
    else:
        assert sample is not None
        candidate_indices = list(sample.candidate_indices)
        candidate_rgb = sample.candidate_rgb_uint8.unsqueeze(0)
        candidate_c2w = sample.candidate_c2w.unsqueeze(0).to(device)
        candidate_k = sample.candidate_intrinsics.unsqueeze(0).to(device)
        candidate_times = sample.candidate_times.unsqueeze(0).to(device)
        episode_ids = sample.candidate_episode_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        memory = model.build_memory_index(
            candidate_rgb,
            candidate_c2w,
            candidate_k,
            candidate_times,
            episode_ids,
            image_hw,
        )
        candidate_camera = model.encode_cameras(
            candidate_c2w,
            candidate_k,
            image_hw,
            candidate_times,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_videos: list[torch.Tensor] = []
    ground_truth_videos: list[torch.Tensor] = []
    traces: list[dict[str, Any]] = []
    generator = torch.Generator(device=device).manual_seed(args.seed)
    latent_h = data_config.height // int(pipe.vae_scale_factor_spatial)
    latent_w = data_config.width // int(pipe.vae_scale_factor_spatial)
    latent_frames = 1 + (data_config.target_rgb_frames - 1) // int(
        pipe.vae_scale_factor_temporal
    )

    for block_index, target_indices in enumerate(target_index_blocks):
        target_c2w = target_c2w_blocks[block_index].unsqueeze(0).to(device)
        target_k = target_k_blocks[block_index].unsqueeze(0).to(device)
        target_times = target_time_blocks[block_index].unsqueeze(0).to(device)
        with torch.no_grad():
            canvas, target_camera = model.target_canvas(
                target_c2w,
                target_k,
                target_times,
                image_hw,
                latent_frames,
            )
            # First run the cheap policy, then VAE-encode only selected source views.
            rollout = model.agent.rollout(
                canvas,
                memory,
                deterministic=True,
                stop_threshold=args.stop_threshold,
            )
            selected_names = [
                (
                    candidate_indices[index]
                    if index < len(candidate_indices)
                    else f"generated_memory_{index - len(candidate_indices)}"
                )
                for index in rollout.selected_views
            ]
            selected_confidences = [
                float(memory.view_confidence[0, index].item())
                for index in rollout.selected_views
            ]
            selected_provenance = [
                "generated"
                if int(memory.view_provenance[0, index].item()) == 1
                else "capture"
                for index in rollout.selected_views
            ]
            source_views = [
                index
                for index in rollout.selected_views
                if index < len(candidate_indices)
            ]
            fine_views = fine_patches = None
            if source_views:
                cpu_indices = torch.tensor(source_views, dtype=torch.long)
                if args.full_capture_memory:
                    selected_internal_indices = [
                        candidate_indices[index] for index in source_views
                    ]
                    selected_rgb = item.read_video_uint8(
                        selected_internal_indices, image_hw
                    ).permute(1, 0, 2, 3).unsqueeze(0)
                else:
                    assert sample is not None
                    selected_rgb = sample.candidate_rgb_uint8.index_select(
                        0, cpu_indices
                    ).unsqueeze(0)
                selected_latents = encode_independent_views(
                    pipe.vae,
                    selected_rgb,
                    device=device,
                    dtype=compute_dtype,
                    chunk_size=config.get("training", {}).get("evidence_vae_chunk", 8),
                )
                with torch.autocast(
                    "cuda",
                    dtype=compute_dtype,
                    enabled=device.type == "cuda"
                    and compute_dtype in {torch.float16, torch.bfloat16},
                ):
                    fine_views, fine_patches = model.encode_fine_latents(
                        selected_latents,
                        candidate_camera.index_select(1, cpu_indices.to(device)),
                    )
            condition = model.make_condition_tokens(
                rollout, memory, target_camera, fine_views, fine_patches
            )
            latents = torch.randn(
                (1, model_config.latent_channels, latent_frames, latent_h, latent_w),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            pipe.scheduler.set_timesteps(
                args.num_inference_steps, device=device, shift=args.shift
            )
            for timestep in pipe.scheduler.timesteps:
                t = timestep.float().reshape(1).to(device)
                with torch.autocast(
                    "cuda",
                    dtype=compute_dtype,
                    enabled=device.type == "cuda"
                    and compute_dtype in {torch.float16, torch.bfloat16},
                ):
                    conditional = model.generator_forward(
                        latents,
                        t,
                        prompt_embeds,
                        prompt_mask,
                        condition,
                    ).float()
                    if args.guidance_scale > 1:
                        unconditional = model.generator_forward(
                            latents,
                            t,
                            negative_embeds,
                            negative_mask,
                            condition,
                            drop_condition=True,
                        ).float()
                    else:
                        unconditional = None
                if args.guidance_scale > 1:
                    assert unconditional is not None
                    prediction = unconditional + args.guidance_scale * (
                        conditional - unconditional
                    )
                else:
                    prediction = conditional
                latents = pipe.scheduler.step(
                    prediction,
                    timestep,
                    latents,
                    generator=generator,
                    return_dict=False,
                )[0]
            decoded = decode_video_latents(
                pipe.vae,
                latents,
                device=device,
                dtype=compute_dtype,
            )[0]
            output_start, output_count = target_output_slices[block_index]
            generated_videos.append(
                decoded[:, output_start : output_start + output_count].cpu()
            )
            if target_rgb_blocks is not None:
                ground_truth_videos.append(
                    target_rgb_blocks[block_index].float().div(255.0)
                )
            if block_index + 1 < len(target_index_blocks):
                latent_indices = model.latent_pose_indices(
                    target_c2w.shape[1], latents.shape[2], device
                )
                update_camera = model.encode_cameras(
                    target_c2w[:, latent_indices],
                    target_k[:, latent_indices],
                    image_hw,
                    target_times[:, latent_indices],
                )
                memory = model.append_latent_memory(
                    memory,
                    latents.permute(0, 2, 1, 3, 4),
                    update_camera,
                    confidence=float(
                        (1.0 - rollout.state.uncertainty.mean())
                        .clamp(0.1, 0.9)
                        .item()
                    ),
                )

        traces.append(
            {
                "block": block_index,
                "target_indices": list(target_indices),
                "selected_memory_indices": selected_names,
                "selected_memory_confidences": selected_confidences,
                "selected_memory_provenance": selected_provenance,
                "query_types": [
                    QUERY_TYPES[value]
                    for value in rollout.query_types
                ],
                "regions": rollout.regions,
                "stop_probabilities": rollout.stop_probabilities,
                "final_uncertainty": float(rollout.state.uncertainty.mean().item()),
            }
        )

    generated = torch.cat(generated_videos, dim=1).permute(1, 2, 3, 0)
    _write_video(output_dir / "generated.mp4", generated, args.fps)
    if ground_truth_videos:
        ground_truth = torch.cat(ground_truth_videos, dim=1).permute(1, 2, 3, 0)
        _write_video(output_dir / "ground_truth.mp4", ground_truth, args.fps)
        comparison = torch.cat((ground_truth, generated), dim=2)
        _write_video(output_dir / "comparison_gt_generated.mp4", comparison, args.fps)
    (output_dir / "retrieval_trace.json").write_text(
        json.dumps(
            {
                "item_name": args.item_name,
                "capture_span": capture_span,
                "target_camera_npz": args.target_camera_npz,
                "candidate_indices": candidate_indices,
                "traces": traces,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info("saved inference outputs to %s", output_dir)


if __name__ == "__main__":
    main()
