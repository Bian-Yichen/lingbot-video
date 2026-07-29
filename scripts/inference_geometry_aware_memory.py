from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.geometry_aware_memory.data import (  # noqa: E402
    GeometryMemorySampleConfig,
    RcloneConfig,
    RoomTourItemCache,
    VipeRoomTourItem,
)
from lingbot_video.geometry_aware_memory.model import (  # noqa: E402
    GIMWorldLingBotModel,
    GIMWorldModelConfig,
)
from lingbot_video.geometry_aware_memory.inference import (  # noqa: E402
    DynamicGIMHistory,
)
from lingbot_video.geometry_aware_memory.pruning import (  # noqa: E402
    MIGreedyPruner,
    PoseTimeKernelConfig,
)
from lingbot_video.geometry_aware_memory.training import (  # noqa: E402
    SceneLatentCache,
    prepare_gim_batch,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GIM-World LingBot checkpoint on one withheld room-tour block."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--item_name", required=True)
    parser.add_argument("--target_start", type=int, default=None)
    parser.add_argument(
        "--dataset_root",
        default="h:bianyichen/AnyReconProDataset_labeled_2/",
    )
    parser.add_argument(
        "--cache_root",
        default="/tmp/lingbot_gim_world_cache",
    )
    parser.add_argument(
        "--latent_cache_root",
        default="/tmp/lingbot_gim_world_latents",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=1,
        help=(
            "Number of consecutive 81-RGB blocks to roll out. Each generated "
            "block is appended to history before rebuilding memory."
        ),
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
    parser.add_argument("--rclone_binary", default="rclone")
    parser.add_argument("--rclone_config", default=None)
    parser.add_argument(
        "--rclone_clear_proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.num_blocks < 1:
        parser.error("--num_blocks must be positive")
    return args


def _checkpoint_file(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "trainable_components.pt"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


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
    model_dir = args.model_dir or training_config.get("model_dir")
    if not model_dir:
        raise ValueError(
            "--model_dir is required because the checkpoint has no model_dir metadata"
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
    # GIM-World explicitly discards the training-only geometry decoder.
    del model.geometry_head
    model.eval().to(device)

    cache = RoomTourItemCache(
        args.dataset_root,
        args.cache_root,
        rclone=RcloneConfig(
            binary=args.rclone_binary,
            config_path=args.rclone_config,
            clear_proxy=args.rclone_clear_proxy,
        ),
    )
    logger.info("materializing scene %s", args.item_name)
    item = VipeRoomTourItem(cache.materialize(args.item_name))
    sample_config = GeometryMemorySampleConfig(
        height=model_config.image_height,
        width=model_config.image_width,
        target_rgb_frames=int(training_config.get("target_rgb_frames", 81)),
        vae_temporal_stride=int(
            training_config.get("vae_temporal_stride", 4)
        ),
        min_memory_rgb_frames=int(
            training_config.get("min_memory_rgb_frames", 800)
        ),
        target_guard_rgb_frames=int(
            training_config.get("target_guard_rgb_frames", 128)
        ),
        samples_per_item=1,
        context_policy=str(
            training_config.get("context_policy", "prefix")
        ),
    )
    if sample_config.context_policy != "prefix" and args.num_blocks > 1:
        raise ValueError(
            "multi-block rollout requires context_policy=prefix; an offline "
            "all_except_target context would contain future ground-truth frames"
        )
    valid_starts = [
        start
        for start in item.valid_target_starts(sample_config)
        if (
            start
            + args.num_blocks * sample_config.target_rgb_frames
            - 1
            <= item.indices[-1]
        )
    ]
    if not valid_starts:
        raise RuntimeError(
            f"{item.root.name} has no target start for {args.num_blocks} "
            "consecutive blocks"
        )
    target_start = (
        random.Random(args.seed).choice(valid_starts)
        if args.target_start is None
        else args.target_start
    )
    if target_start not in valid_starts:
        raise ValueError(
            f"target_start={target_start} cannot support {args.num_blocks} "
            f"blocks; examples of valid starts: {valid_starts[:8]}"
        )
    sample = item.make_sample(
        sample_config,
        random.Random(args.seed),
        target_start=target_start,
    )
    pruner = MIGreedyPruner(
        PoseTimeKernelConfig(
            sigma_position=float(training_config.get("sigma_position", -1.0)),
            sigma_rotation=float(
                training_config.get("sigma_rotation", np.pi / 6)
            ),
            sigma_time=float(training_config.get("sigma_time", 50.0)),
            jitter=float(training_config.get("kernel_jitter", 1e-5)),
        )
    )
    latent_cache = SceneLatentCache(
        args.latent_cache_root,
        height=model_config.image_height,
        width=model_config.image_width,
        temporal_stride=sample_config.vae_temporal_stride,
        chunk_rgb_frames=int(
            training_config.get("vae_encode_chunk_rgb_frames", 81)
        ),
    )
    prepared = prepare_gim_batch(
        sample,
        vae=pipe.vae,
        latent_cache=latent_cache,
        pruner=pruner,
        pruning_budget=int(training_config.get("pruning_budget", 200)),
        vae_temporal_stride=sample_config.vae_temporal_stride,
        device=device,
        compute_dtype=compute_dtype,
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

    generator = torch.Generator(device=device).manual_seed(args.seed)
    history_state = DynamicGIMHistory(
        latents=prepared["all_history_latents"],
        c2w=prepared["all_history_c2w"],
        intrinsics=prepared["all_history_intrinsics"],
        times=prepared["all_history_times"],
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_blocks: list[np.ndarray] = []
    ground_truth_blocks: list[np.ndarray] = []
    block_metadata: list[dict] = []
    latent_shape = tuple(prepared["target_latents"].shape)
    pruning_budget = int(training_config.get("pruning_budget", 200))
    autocast = (
        torch.autocast("cuda", dtype=compute_dtype)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )
    with torch.no_grad():
        for block_index in range(args.num_blocks):
            block_start = (
                sample.target_start
                + block_index * sample_config.target_rgb_frames
            )
            block_rgb_indices = tuple(
                range(
                    block_start,
                    block_start + sample_config.target_rgb_frames,
                )
            )
            block_latent_indices = block_rgb_indices[
                :: sample_config.vae_temporal_stride
            ]
            target_c2w, target_intrinsics = item.cameras(
                block_latent_indices,
                sample.image_hw,
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
            history_state.append(
                latents.detach().cpu().to(
                    prepared["all_history_latents"].dtype
                ),
                target_c2w.cpu(),
                target_intrinsics.cpu(),
                torch.tensor(
                    block_latent_indices,
                    dtype=history_state.times.dtype,
                ),
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
                    "target_start": block_start,
                    "target_rgb_indices": list(block_rgb_indices),
                    "history_latent_frames_before": history_before,
                    "retained_history_times": [
                        int(value) for value in retained_times
                    ],
                    "history_latent_frames_after": (
                        history_state.frame_count
                    ),
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
        "target_start": sample.target_start,
        "target_rgb_indices": list(sample.target_rgb_indices),
        "context_policy": sample_config.context_policy,
        "memory_rgb_frames_before_vae": len(sample.memory_rgb_indices),
        "memory_latent_candidates": int(
            prepared["history_candidates"].item()
        ),
        "memory_latent_retained": int(
            prepared["history_retained"].item()
        ),
        "updated_history_latent_frames": history_state.frame_count,
        "num_blocks": args.num_blocks,
        "blocks": block_metadata,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "paper_inference_note": (
            "VGGT and the camera-query geometry head are intentionally absent "
            "at inference, as specified by GIM-World."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("saved inference outputs to %s", output_dir)


if __name__ == "__main__":
    main()
