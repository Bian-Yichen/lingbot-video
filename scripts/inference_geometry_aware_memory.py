from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as iio_v2
import imageio.v3 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.geometry_aware_memory.data import (  # noqa: E402
    GeometryMemorySampleConfig,
    LocalRoomTourIndex,
    SOURCE_FRAME_STRIDE,
    VipeRoomTourItem,
)
from lingbot_video.geometry_aware_memory.inference import (  # noqa: E402
    DynamicGIMHistory,
)
from lingbot_video.geometry_aware_memory.lora import (  # noqa: E402
    inject_backbone_lora,
    lora_config_from_mapping,
    validate_partial_checkpoint_load,
)
from lingbot_video.geometry_aware_memory.model import (  # noqa: E402
    GIMWorldLingBotModel,
    GIMWorldModelConfig,
)
from lingbot_video.geometry_aware_memory.pruning import (  # noqa: E402
    MIGreedyPruner,
    PoseTimeKernelConfig,
)
from lingbot_video.geometry_aware_memory.training import (  # noqa: E402
    encode_wan_frames_independently,
)
from lingbot_video.pipeline_lingbot_video import (  # noqa: E402
    DEFAULT_NEGATIVE_PROMPT,
    LingBotVideoPipeline,
)
from lingbot_video.runner import _patch_qwen3vl_from_pretrained  # noqa: E402
from lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModel,
)


logger = logging.getLogger("lingbot_video.inference_geometry_aware_memory")


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
            "Generate a continuous target trajectory from pose-retrieved "
            "local memory views."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--item_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--local_window_start", type=int, default=None)
    parser.add_argument("--target_start", type=int, default=None)
    parser.add_argument("--memory_view_count", type=int, default=None)
    parser.add_argument("--sample_epoch", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=1,
        help="Each block generates one target_rgb_frames clip and writes it back.",
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument(
        "--save_capture_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the exact sparse capture sequence supplied to memory.",
    )
    parser.add_argument(
        "--mixed_precision",
        choices=["fp16", "bf16"],
        default="bf16",
    )
    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    for name in ("checkpoint", "item_name", "dataset_root", "output_dir"):
        if not getattr(args, name):
            parser.error(f"--{name} is required (it may be supplied by --config)")
    if args.num_blocks < 1:
        parser.error("--num_blocks must be positive")
    explicit = (
        args.local_window_start,
        args.target_start,
        args.memory_view_count,
    )
    if any(value is not None for value in explicit) and not all(
        value is not None for value in explicit
    ):
        parser.error(
            "explicit sampling requires --local_window_start, "
            "--target_start, and --memory_view_count together"
        )
    if args.negative_prompt is None:
        args.negative_prompt = DEFAULT_NEGATIVE_PROMPT
    return args


def _checkpoint_file(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "trainable_components.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _dtype(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]


def _save_video(path: Path, frames: np.ndarray, fps: float) -> None:
    frames = np.clip(frames * 255.0, 0, 255).astype(np.uint8)
    iio.imwrite(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    )


def _save_indexed_video(
    path: Path,
    item: VipeRoomTourItem,
    indices: tuple[int, ...],
    target_hw: tuple[int, int],
    fps: float,
) -> None:
    """Stream a potentially long capture video without holding it in RAM."""
    with iio_v2.get_writer(
        path,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    ) as writer:
        for index in indices:
            frame = (
                item.read_rgb(index, target_hw)
                .permute(1, 2, 0)
                .mul(255.0)
                .clamp_(0, 255)
                .byte()
                .numpy()
            )
            writer.append_data(frame)


@torch.no_grad()
def _decode_wan_frames_independently(
    pipe: LingBotVideoPipeline,
    latents: torch.Tensor,
    *,
    chunk_frames: int,
) -> np.ndarray:
    """Decode [1,C,T,H,W] as T unrelated one-frame VAE samples."""

    if latents.shape[0] != 1:
        raise ValueError("independent inference decoding currently expects B=1")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    flat = latents.permute(0, 2, 1, 3, 4).reshape(
        latents.shape[2],
        latents.shape[1],
        1,
        latents.shape[3],
        latents.shape[4],
    )
    frames: list[np.ndarray] = []
    for start in range(0, flat.shape[0], chunk_frames):
        decoded = pipe._decode_latents(flat[start : start + chunk_frames])
        frames.extend(video[0] for video in decoded)
    return np.stack(frames, axis=0)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        # Current checkpoints also contain AdamW state. mmap avoids eagerly
        # copying the entire multi-GB archive into anonymous CPU memory.
        checkpoint = torch.load(path, mmap=True, **kwargs)
    except TypeError:
        checkpoint = torch.load(path, **kwargs)
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain one dictionary")
    for required in ("model", "model_config", "training_config"):
        if required not in checkpoint:
            raise KeyError(f"checkpoint is missing required key {required!r}")
    # The optimizer is irrelevant for inference and can otherwise keep many
    # GB of mapped tensors alive in the Python object graph.
    checkpoint.pop("optimizer", None)
    return checkpoint


def _sample_epoch_from_checkpoint(
    checkpoint: dict[str, Any],
    requested_epoch: int | None,
) -> int:
    if requested_epoch is not None:
        if requested_epoch < 0:
            raise ValueError("sample_epoch cannot be negative")
        return int(requested_epoch)
    # Checkpoints are saved after an epoch with next_epoch pointing at the
    # following zero-based epoch. Reuse the most recently trained curriculum.
    return max(int(checkpoint.get("next_epoch", 1)) - 1, 0)


def _validate_loaded_state(
    missing: list[str],
    unexpected: list[str],
    *,
    backbone_train_mode: str,
) -> None:
    validate_partial_checkpoint_load(
        missing,
        unexpected,
        backbone_train_mode=backbone_train_mode,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    checkpoint_path = _checkpoint_file(args.checkpoint)
    logger.info("loading checkpoint %s", checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    training_config = checkpoint.get("training_config", {})
    if training_config.get("vae_frame_mode") != "independent":
        raise ValueError(
            "checkpoint was not trained with independent one-frame VAE "
            "latents and the local retrieval sampler; use the inference code "
            "from that checkpoint's training commit"
        )
    model_dir = args.model_dir or training_config.get("model_dir")
    if not model_dir:
        raise ValueError(
            "--model_dir is required because checkpoint metadata has no model_dir"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = _dtype(args.mixed_precision)
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        model_dir,
        subfolder="transformer",
        torch_dtype=compute_dtype,
    )
    patch_context = (
        _patch_qwen3vl_from_pretrained()
        if _patch_qwen3vl_from_pretrained is not None
        else contextlib.nullcontext()
    )
    with patch_context:
        pipe = LingBotVideoPipeline.from_pretrained(
            model_dir,
            transformer=transformer,
            trust_remote_code=True,
            torch_dtype={
                "default": compute_dtype,
                "transformer": compute_dtype,
                "text_encoder": compute_dtype,
                "vae": torch.float32,
            },
        )
    pipe.to(device)
    pipe.vae.requires_grad_(False).eval()
    backbone_train_mode = str(
        training_config.get("backbone_train_mode", "full")
    )
    lora_summary = None
    if backbone_train_mode == "lora":
        lora_summary = inject_backbone_lora(
            pipe.transformer,
            lora_config_from_mapping(training_config),
        )
    model_config = GIMWorldModelConfig(**checkpoint["model_config"])
    model = GIMWorldLingBotModel(pipe.transformer, model_config)
    checkpoint_model = checkpoint.pop("model")
    missing, unexpected = model.load_state_dict(
        checkpoint_model,
        strict=False,
    )
    del checkpoint_model
    gc.collect()
    _validate_loaded_state(
        missing,
        unexpected,
        backbone_train_mode=backbone_train_mode,
    )
    if missing:
        logger.warning(
            "loaded %s-backbone checkpoint; restored %d omitted frozen "
            "backbone tensors from model_dir",
            backbone_train_mode,
            len(missing),
        )
    if lora_summary is not None:
        logger.info(
            "loaded backbone LoRA modules=%d trainable_parameters=%d rank=%d",
            lora_summary.module_count,
            lora_summary.parameter_count,
            int(training_config.get("lora_rank", 32)),
        )
    model.eval().to(device)

    item_path = LocalRoomTourIndex(args.dataset_root).item_path(args.item_name)
    item = VipeRoomTourItem(item_path)
    trained_query_blocks = int(training_config.get("query_blocks", 1))
    if args.num_blocks != trained_query_blocks:
        logger.warning(
            "checkpoint trained with query_blocks=%d but inference requested "
            "num_blocks=%d; blocks after the first use generated-latent memory "
            "and are rollout evaluation, not an input exactly seen in training",
            trained_query_blocks,
            args.num_blocks,
        )
    sample_config = GeometryMemorySampleConfig(
        height=model_config.image_height,
        width=model_config.image_width,
        target_rgb_frames=int(
            training_config.get("target_rgb_frames", 41)
        ),
        query_blocks=args.num_blocks,
        local_window_rgb_frames=int(
            training_config.get("local_window_rgb_frames", 81)
        ),
        memory_views_min=int(
            training_config.get("memory_views_min", 2)
        ),
        memory_views_max=int(
            training_config.get("memory_views_max", 24)
        ),
        retrieval_rotation_weight=float(
            training_config.get("retrieval_rotation_weight", 0.25)
        ),
        retrieval_temperature=float(
            training_config.get("retrieval_temperature", 0.25)
        ),
    )
    sample_config.validate()
    sample_epoch = _sample_epoch_from_checkpoint(
        checkpoint,
        args.sample_epoch,
    )
    sample = item.make_sample(
        sample_config,
        random.Random(args.seed),
        epoch=sample_epoch,
        local_window_start=args.local_window_start,
        target_start=args.target_start,
        memory_view_count=args.memory_view_count,
    )
    logger.info(
        "scene=%s sample_epoch=%d window=%d:%d memory_views=%d "
        "target=%d:%d (%d blocks) retrieval_coverage=%.4f "
        "pose_cost=%.4f",
        args.item_name,
        sample_epoch,
        sample.local_window_start,
        sample.local_window_end,
        len(sample.capture_rgb_indices),
        sample.query_start,
        sample.query_rgb_blocks[-1][-1],
        args.num_blocks,
        sample.retrieval_coverage_score,
        sample.trajectory_overlap_score,
    )

    # Inference input contains capture RGB only. Query RGB is never encoded;
    # it is read after generation solely for evaluation videos.
    logger.info(
        "encoding %d capture RGB frames with the online Wan VAE",
        len(sample.capture_rgb_indices),
    )
    capture_latents = encode_wan_frames_independently(
        pipe.vae,
        item,
        sample.capture_rgb_indices,
        sample.image_hw,
        read_chunk_rgb_frames=int(
            training_config.get("vae_encode_chunk_rgb_frames", 4)
        ),
        device=device,
        dtype=compute_dtype,
    )
    capture_latent_indices = sample.capture_rgb_indices
    capture_c2w, capture_intrinsics = item.cameras(
        capture_latent_indices,
        sample.image_hw,
        origin_index=sample.capture_start,
    )
    capture_times = torch.tensor(
        [
            int(index) - sample.capture_start
            for index in capture_latent_indices
        ],
        dtype=torch.long,
    )
    history_state = DynamicGIMHistory(
        latents=capture_latents,
        c2w=capture_c2w.unsqueeze(0),
        intrinsics=capture_intrinsics.unsqueeze(0),
        times=capture_times,
    )

    prompt = args.prompt or training_config.get(
        "prompt",
        "An indoor room tour.",
    )
    prompt_embeds, prompt_mask = pipe.encode_prompt(prompt, device=device)
    do_cfg = args.guidance_scale > 1.0
    if do_cfg:
        negative_embeds, negative_mask = pipe.encode_prompt(
            args.negative_prompt,
            device=device,
        )
    pipe.text_encoder.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pruner = MIGreedyPruner(
        PoseTimeKernelConfig(
            sigma_position=float(
                training_config.get("sigma_position", -1.0)
            ),
            sigma_rotation=float(
                training_config.get("sigma_rotation", np.pi / 6)
            ),
            sigma_time=float(training_config.get("sigma_time", 50.0)),
            jitter=float(training_config.get("kernel_jitter", 1e-5)),
        )
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    generated_blocks: list[np.ndarray] = []
    ground_truth_blocks: list[np.ndarray] = []
    block_metadata: list[dict[str, Any]] = []
    latent_shape = (
        1,
        capture_latents.shape[1],
        sample_config.target_latent_frames,
        capture_latents.shape[3],
        capture_latents.shape[4],
    )
    pruning_budget = int(training_config.get("pruning_budget", 200))
    autocast = (
        torch.autocast("cuda", dtype=compute_dtype)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )

    with torch.no_grad():
        for block_index, block_rgb_indices in enumerate(
            sample.query_rgb_blocks
        ):
            block_latent_indices = block_rgb_indices
            target_c2w, target_intrinsics = item.cameras(
                block_latent_indices,
                sample.image_hw,
                origin_index=sample.capture_start,
            )
            target_c2w = target_c2w.unsqueeze(0).to(device)
            target_intrinsics = target_intrinsics.unsqueeze(0).to(device)
            history_before = history_state.frame_count
            with autocast:
                memory, retained_positions = history_state.build_memory(
                    model,
                    pruner,
                    budget=pruning_budget,
                    device=device,
                    dtype=compute_dtype,
                )
                actions = model.target_action_embeddings(
                    target_c2w,
                    target_intrinsics,
                )
            logger.info(
                "block=%d/%d history=%d retained=%d memory_shape=%s "
                "target_latent_frames=%d",
                block_index + 1,
                len(sample.query_rgb_blocks),
                history_before,
                int(retained_positions.numel()),
                tuple(memory.shape),
                len(block_latent_indices),
            )

            latents = torch.randn(
                latent_shape,
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            pipe.scheduler.set_timesteps(
                args.num_inference_steps,
                device=device,
                shift=args.flow_shift,
            )
            for timestep in pipe.progress_bar(pipe.scheduler.timesteps):
                timestep_batch = timestep.expand(latents.shape[0]).to(device)
                with autocast:
                    conditional = model.denoise(
                        latents,
                        timestep_batch,
                        prompt_embeds.to(compute_dtype),
                        memory=memory,
                        target_action_embeddings=actions,
                        encoder_attention_mask=prompt_mask,
                    ).float()
                    if do_cfg:
                        unconditional = model.denoise(
                            latents,
                            timestep_batch,
                            negative_embeds.to(compute_dtype),
                            memory=memory,
                            target_action_embeddings=actions,
                            encoder_attention_mask=negative_mask,
                        ).float()
                        conditional = (
                            unconditional
                            + args.guidance_scale
                            * (conditional - unconditional)
                        )
                latents = pipe.scheduler.step(
                    conditional,
                    timestep,
                    latents,
                    return_dict=False,
                    generator=generator,
                )[0]

            retained_times = history_state.times[retained_positions].tolist()
            update_times = torch.tensor(
                [
                    int(index) - sample.capture_start
                    for index in block_latent_indices
                ],
                dtype=history_state.times.dtype,
            )
            history_state.append(
                latents.detach().cpu().to(capture_latents.dtype),
                target_c2w.cpu(),
                target_intrinsics.cpu(),
                update_times,
            )

            generated = _decode_wan_frames_independently(
                pipe,
                latents,
                chunk_frames=int(
                    training_config.get("vae_encode_chunk_rgb_frames", 4)
                ),
            )
            ground_truth = (
                item.read_video(block_rgb_indices, sample.image_hw)
                .permute(1, 2, 3, 0)
                .numpy()
            )
            generated_blocks.append(generated)
            ground_truth_blocks.append(ground_truth)
            _save_video(
                output_dir / f"generated_block_{block_index:03d}.mp4",
                generated,
                args.fps,
            )
            _save_video(
                output_dir / f"ground_truth_block_{block_index:03d}.mp4",
                ground_truth,
                args.fps,
            )
            _save_video(
                output_dir
                / f"comparison_gt_generated_block_{block_index:03d}.mp4",
                np.concatenate((ground_truth, generated), axis=2),
                args.fps,
            )
            block_metadata.append(
                {
                    "block_index": block_index,
                    "query_rgb_indices": list(block_rgb_indices),
                    "query_latent_source_indices": list(block_latent_indices),
                    "history_latent_frames_before": history_before,
                    "retained_history_times": [
                        int(value) for value in retained_times
                    ],
                    "appended_model_times": update_times.tolist(),
                    "history_latent_frames_after": history_state.frame_count,
                }
            )
            del memory, actions, latents

    generated = np.concatenate(generated_blocks, axis=0)
    ground_truth = np.concatenate(ground_truth_blocks, axis=0)
    _save_video(output_dir / "generated.mp4", generated, args.fps)
    _save_video(output_dir / "ground_truth.mp4", ground_truth, args.fps)
    _save_video(
        output_dir / "comparison_gt_generated.mp4",
        np.concatenate((ground_truth, generated), axis=2),
        args.fps,
    )
    if args.save_capture_video:
        logger.info("saving the sparse capture conditioning video")
        _save_indexed_video(
            output_dir / "conditioning_capture.mp4",
            item,
            sample.capture_rgb_indices,
            sample.image_hw,
            args.fps,
        )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "item_name": args.item_name,
        "dataset_root": args.dataset_root,
        "local_window_internal_range": [
            sample.local_window_start,
            sample.local_window_end,
        ],
        "memory_view_count": len(sample.capture_rgb_indices),
        "capture_rgb_indices": list(sample.capture_rgb_indices),
        "capture_source_rgb_indices": [
            int(index) * SOURCE_FRAME_STRIDE
            for index in sample.capture_rgb_indices
        ],
        "query_rgb_blocks": [
            list(block) for block in sample.query_rgb_blocks
        ],
        "query_source_rgb_blocks": [
            [
                int(index) * SOURCE_FRAME_STRIDE
                for index in block
            ]
            for block in sample.query_rgb_blocks
        ],
        "capture_origin_internal_index": sample.capture_start,
        "capture_origin_source_rgb_index": (
            sample.capture_start * SOURCE_FRAME_STRIDE
        ),
        "sample_epoch": sample_epoch,
        "checkpoint_next_epoch": int(checkpoint.get("next_epoch", 0)),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "trajectory_overlap_score": sample.trajectory_overlap_score,
        "retrieval_coverage_score": sample.retrieval_coverage_score,
        "initial_capture_latent_frames": capture_latents.shape[2],
        "updated_history_latent_frames": history_state.frame_count,
        "num_blocks": args.num_blocks,
        "blocks": block_metadata,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "resolution": list(sample.image_hw),
        "mixed_precision": args.mixed_precision,
        "save_capture_video": args.save_capture_video,
        "vae_frame_mode": "independent one-frame encode/decode",
        "cache_mode": "disabled; memory-view VAE runs online from RGB",
        "query_input_note": (
            "Only query camera poses/intrinsics enter generation. Query RGB is "
            "read after generation for ground-truth evaluation only."
        ),
        "dynamic_update_note": (
            "Each generated latent block is appended and memory is rebuilt "
            "before the next block."
        ),
        "paper_inference_note": (
            "VGGT and the camera-query geometry head are discarded at inference."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("saved inference outputs to %s", output_dir)


if __name__ == "__main__":
    main()
