from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.geometry_aware_memory.data import (  # noqa: E402
    GeometryMemorySampleConfig,
    LocalRoomTourIndex,
    VipeRoomTourItem,
)
from lingbot_video.geometry_aware_memory.inference import (  # noqa: E402
    DynamicGIMHistory,
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
    encode_wan_scene_streaming,
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
            "Roll out a withheld query phase from one sparse capture window."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--item_name", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--capture_start", type=int, default=None)
    parser.add_argument("--query_start", type=int, default=None)
    parser.add_argument("--capture_rgb_frames", type=int, default=None)
    parser.add_argument("--capture_frame_stride", type=int, default=None)
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
        args.capture_start,
        args.query_start,
        args.capture_rgb_frames,
        args.capture_frame_stride,
    )
    if any(value is not None for value in explicit) and not all(
        value is not None for value in explicit
    ):
        parser.error(
            "explicit sampling requires --capture_start, --query_start, "
            "--capture_rgb_frames, and --capture_frame_stride together"
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


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    checkpoint_path = _checkpoint_file(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    training_config = checkpoint.get("training_config", {})
    if "capture_window_min_rgb_frames" not in training_config:
        raise ValueError(
            "checkpoint was trained with the previous extrapolative sampler; "
            "it is not data-compatible with interleaved capture/query "
            "inference. Retrain from this branch or run the checkpoint from "
            "the earlier commit."
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
    model_config = GIMWorldModelConfig(**checkpoint["model_config"])
    model = GIMWorldLingBotModel(pipe.transformer, model_config)
    missing, unexpected = model.load_state_dict(
        checkpoint["model"],
        strict=False,
    )
    allowed_missing = (
        training_config.get("backbone_train_mode") == "frozen"
        and all(name.startswith("backbone.") for name in missing)
    )
    if unexpected or (missing and not allowed_missing):
        logger.warning(
            "checkpoint load missing=%d unexpected=%d",
            len(missing),
            len(unexpected),
        )
    # VGGT and the geometry decoder are training-only in GIM-World.
    del model.geometry_head
    model.eval().to(device)

    item_path = LocalRoomTourIndex(args.dataset_root).item_path(args.item_name)
    item = VipeRoomTourItem(item_path)
    sample_config = GeometryMemorySampleConfig(
        height=model_config.image_height,
        width=model_config.image_width,
        target_rgb_frames=int(
            training_config.get("target_rgb_frames", 49)
        ),
        query_blocks=args.num_blocks,
        vae_temporal_stride=int(
            training_config.get("vae_temporal_stride", 4)
        ),
        capture_window_min_rgb_frames=int(
            training_config.get("capture_window_min_rgb_frames", 257)
        ),
        capture_window_max_rgb_frames=int(
            training_config.get("capture_window_max_rgb_frames", 1000)
        ),
        capture_window_curriculum_start_max_rgb_frames=int(
            training_config.get(
                "capture_window_curriculum_start_max_rgb_frames",
                321,
            )
        ),
        capture_window_curriculum_epochs=int(
            training_config.get(
                "capture_window_curriculum_epochs",
                5,
            )
        ),
        capture_window_min_fraction_of_current_max=float(
            training_config.get(
                "capture_window_min_fraction_of_current_max",
                0.75,
            )
        ),
        capture_frame_stride_min=int(
            training_config.get("capture_frame_stride_min", 2)
        ),
        capture_frame_stride_max=int(
            training_config.get("capture_frame_stride_max", 3)
        ),
        trajectory_pose_stride=int(
            training_config.get("trajectory_pose_stride", 4)
        ),
        trajectory_rotation_weight=float(
            training_config.get("trajectory_rotation_weight", 0.25)
        ),
    )
    sample_epoch = (
        max(sample_config.capture_window_curriculum_epochs - 1, 0)
        if args.sample_epoch is None
        else args.sample_epoch
    )
    sample = item.make_sample(
        sample_config,
        random.Random(args.seed),
        epoch=sample_epoch,
        capture_start=args.capture_start,
        query_start=args.query_start,
        capture_rgb_frames=args.capture_rgb_frames,
        capture_frame_stride=args.capture_frame_stride,
    )
    logger.info(
        "scene=%s window=%d:%d stride=%d capture=%d RGB "
        "query=%d:%d (%d blocks) "
        "coverage_score=%.4f",
        args.item_name,
        sample.capture_window_start,
        sample.capture_window_end,
        sample.capture_frame_stride,
        len(sample.capture_rgb_indices),
        sample.query_start,
        sample.query_rgb_blocks[-1][-1],
        args.num_blocks,
        sample.trajectory_overlap_score,
    )

    # Inference input contains capture RGB only. Query RGB is never encoded;
    # it is read after generation solely for evaluation videos.
    capture_latents = encode_wan_scene_streaming(
        pipe.vae,
        item,
        sample.capture_rgb_indices,
        sample.image_hw,
        temporal_stride=sample_config.vae_temporal_stride,
        read_chunk_rgb_frames=int(
            training_config.get("vae_encode_chunk_rgb_frames", 81)
        ),
        device=device,
        dtype=compute_dtype,
    )
    capture_latent_indices = sample.capture_rgb_indices[
        :: sample_config.vae_temporal_stride
    ]
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
            block_latent_indices = block_rgb_indices[
                :: sample_config.vae_temporal_stride
            ]
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
            next_time = (
                int(history_state.times.max().item())
                + sample.capture_frame_stride
                * sample_config.vae_temporal_stride
            )
            update_times = (
                torch.arange(
                    latents.shape[2],
                    dtype=history_state.times.dtype,
                )
                * sample.capture_frame_stride
                * sample_config.vae_temporal_stride
                + next_time
            )
            history_state.append(
                latents.detach().cpu().to(capture_latents.dtype),
                target_c2w.cpu(),
                target_intrinsics.cpu(),
                update_times,
            )

            generated = pipe._decode_latents(latents)[0]
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
    metadata = {
        "checkpoint": str(checkpoint_path),
        "item_name": args.item_name,
        "dataset_root": args.dataset_root,
        "capture_window_internal_range": [
            sample.capture_window_start,
            sample.capture_window_end,
        ],
        "capture_frame_stride": sample.capture_frame_stride,
        "query_phase_offset": sample.query_phase_offset,
        "capture_rgb_indices": list(sample.capture_rgb_indices),
        "query_rgb_blocks": [
            list(block) for block in sample.query_rgb_blocks
        ],
        "capture_origin_source_internal_index": sample.capture_start,
        "trajectory_overlap_score": sample.trajectory_overlap_score,
        "initial_capture_latent_frames": capture_latents.shape[2],
        "updated_history_latent_frames": history_state.frame_count,
        "num_blocks": args.num_blocks,
        "blocks": block_metadata,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "cache_mode": "disabled; capture VAE runs online from RGB",
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
