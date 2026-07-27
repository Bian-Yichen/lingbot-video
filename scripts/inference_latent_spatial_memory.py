from __future__ import annotations

import argparse
import contextlib
import gc
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.latent_spatial_memory.controlnet import (  # noqa: E402
    LingBotLatentMemoryControlNet,
)
from lingbot_video.latent_spatial_memory.data import (  # noqa: E402
    RcloneConfig,
    RoomTourItemCache,
    VipeRoomTourItem,
)
from lingbot_video.latent_spatial_memory.geometry import (  # noqa: E402
    depth_validity_mask,
    downsample_depth_bilinear,
    intrinsics_vector_to_matrix,
    make_plucker_rays,
    normalize_c2w_to_first_capture,
    resize_crop_intrinsics,
    scale_intrinsics,
)
from lingbot_video.latent_spatial_memory.memory import (  # noqa: E402
    LatentSpatialMemory,
    memory_consistency_mask,
)
from lingbot_video.latent_spatial_memory.model import (  # noqa: E402
    LatentMetricDepthHead,
    LingBotVideoLatentMemoryModel,
)
from lingbot_video.latent_spatial_memory.training import (  # noqa: E402
    encode_capture_latents,
)
from lingbot_video.pipeline_lingbot_video import (  # noqa: E402
    LingBotVideoPipeline,
    _transformer_autocast,
    _transformer_timestep,
)
from lingbot_video.runner import _patch_qwen3vl_from_pretrained  # noqa: E402
from lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset_root",
        default="h:bianyichen/AnyReconProDataset_labeled/",
    )
    parser.add_argument("--item_name", required=True)
    parser.add_argument("--cache_root", default="/tmp/lingbot_latent_memory_cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_start", type=int, required=True)
    parser.add_argument("--history_frames", type=int, default=4096)
    parser.add_argument("--capture_clips", type=int, default=20)
    parser.add_argument("--capture_clip_rgb_frames", type=int, default=9)
    parser.add_argument("--num_frames", type=int, default=257)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--chunk_latent_frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--prompt", default="An indoor room tour.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--memory_max_points", type=int, default=2_000_000)
    parser.add_argument("--memory_voxel_size", type=float, default=0.02)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--depth_edge_threshold", type=float, default=0.08)
    parser.add_argument("--memory_consistency_threshold", type=float, default=0.15)
    parser.add_argument("--rclone_config", default=None)
    parser.add_argument("--rclone_clear_proxy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _select_capture_clips(
    item: VipeRoomTourItem,
    target_start: int,
    history_frames: int,
    count: int,
    clip_frames: int,
) -> tuple[list[int], list[int]]:
    if clip_frames < 1 or (clip_frames - 1) % 4:
        raise ValueError("capture_clip_rgb_frames must equal 1 + 4k")
    rgb = set(item.rgb_by_index)
    geometry = (
        set(item.depth_location)
        & set(item.pose_by_index)
        & set(item.intrinsics_by_index)
    )
    anchor_offsets = tuple(range(0, clip_frames, 4))
    first_start = max(min(rgb), target_start - history_frames)
    final_start = target_start - clip_frames + 1
    candidates = []
    for start in range(first_start, final_start + 1):
        if not all(start + offset in rgb for offset in range(clip_frames)):
            continue
        if not all(start + offset in geometry for offset in anchor_offsets):
            continue
        candidates.append(start)
    if final_start not in candidates:
        raise ValueError("the causal capture clip ending at target_start is incomplete")
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} valid capture clips, requested {count}")
    positions = np.linspace(0, len(candidates) - 1, count).round().astype(np.int64)
    selected = sorted(set(candidates[int(position)] for position in positions))
    for start in candidates:
        if len(selected) == count:
            break
        if start not in selected:
            selected.append(start)
    selected = sorted(selected)[:count]
    if selected[-1] != final_start:
        selected[-1] = final_start
        selected = sorted(selected)
    anchor_indices = [
        start + offset
        for start in selected
        for offset in anchor_offsets
    ]
    return selected, anchor_indices


def _load_model(args: argparse.Namespace, device: torch.device):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stage = payload.get("stage", "side_branch")
    indices = tuple(
        int(value)
        for value in payload.get("control_block_indices", "0,3,6,9,12,15,18,21").split(",")
    )
    backbone = LingBotVideoTransformer3DModel.from_pretrained(
        args.model_dir,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    latent_channels = int(backbone.config.in_channels)
    controlnet = LingBotLatentMemoryControlNet.from_backbone(backbone, indices)
    if stage == "lora":
        from peft import LoraConfig, get_peft_model

        backbone = get_peft_model(
            backbone,
            LoraConfig(
                r=int(payload.get("lora_rank", 64)),
                lora_alpha=int(payload.get("lora_alpha", 64)),
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
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    trainable_missing = [
        name
        for name in missing
        if name.startswith(("controlnet.", "depth_head."))
        or "lora_" in name
    ]
    if trainable_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing_trainable={trainable_missing}, unexpected={unexpected}"
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
                "default": torch.bfloat16,
                "transformer": torch.bfloat16,
                "text_encoder": torch.bfloat16,
                "vae": torch.float32,
            },
        )
    pipe.to(device)
    model.to(device).eval()
    return model, pipe


def main() -> None:
    args = parse_args()
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must both be multiples of 16")
    if args.chunk_latent_frames < 2:
        raise ValueError("chunk_latent_frames must be at least 2")
    target_latent_count = (args.num_frames - 1) // 4 + 1
    if args.num_frames < 1 or (args.num_frames - 1) % 4:
        raise ValueError(
            "num_frames must be 4n+1 because LingBot's VAE temporal stride is 4"
        )
    if (
        target_latent_count < args.chunk_latent_frames
        or (target_latent_count - 1) % (args.chunk_latent_frames - 1)
    ):
        rgb_stride = 4 * (args.chunk_latent_frames - 1)
        raise ValueError(
            f"num_frames must be {rgb_stride}n+1 and contain at least one chunk"
        )
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    cache = RoomTourItemCache(
        args.dataset_root,
        args.cache_root,
        rclone=RcloneConfig(
            config_path=args.rclone_config,
            clear_proxy=args.rclone_clear_proxy,
        ),
    )
    item = VipeRoomTourItem(cache.materialize(args.item_name))
    capture_clip_starts, capture_indices = _select_capture_clips(
        item,
        args.target_start,
        args.history_frames,
        args.capture_clips,
        args.capture_clip_rgb_frames,
    )
    target_indices = [
        args.target_start + 4 * index for index in range(target_latent_count)
    ]
    missing_target = [
        index
        for index in target_indices
        if index not in item.pose_by_index or index not in item.intrinsics_by_index
    ]
    if missing_target:
        raise ValueError(f"target trajectory misses calibration frames: {missing_target[:8]}")

    target_hw = (args.height, args.width)
    source_hw = item.source_hw
    capture_rgb = torch.stack(
        [
            torch.stack(
                [
                    item.read_rgb(start + offset, target_hw)
                    for offset in range(args.capture_clip_rgb_frames)
                ],
                dim=1,
            )
            for start in capture_clip_starts
        ]
    ).unsqueeze(0).to(device)
    capture_depth = torch.stack(
        [item.read_depth(index, target_hw) for index in capture_indices]
    ).to(device)
    capture_c2w = torch.from_numpy(
        np.stack([item.pose_by_index[index] for index in capture_indices])
    ).float().to(device)
    target_c2w = torch.from_numpy(
        np.stack([item.pose_by_index[index] for index in target_indices])
    ).float().to(device)
    first_capture = capture_c2w[0]
    capture_c2w = normalize_c2w_to_first_capture(capture_c2w, first_capture)
    target_c2w = normalize_c2w_to_first_capture(target_c2w, first_capture)
    capture_k = intrinsics_vector_to_matrix(
        resize_crop_intrinsics(
            torch.from_numpy(
                np.stack([item.intrinsics_by_index[index] for index in capture_indices])
            ).float().to(device),
            source_hw,
            target_hw,
        )
    )
    target_k = intrinsics_vector_to_matrix(
        resize_crop_intrinsics(
            torch.from_numpy(
                np.stack([item.intrinsics_by_index[index] for index in target_indices])
            ).float().to(device),
            source_hw,
            target_hw,
        )
    )

    model, pipe = _load_model(args, device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        capture_latents = encode_capture_latents(
            pipe.vae,
            capture_rgb,
            micro_batch_size=2,
            generator=generator,
        )[0]
    latent_hw = (capture_latents.shape[-2], capture_latents.shape[-1])
    capture_k_latent = scale_intrinsics(capture_k, target_hw, latent_hw)
    target_k_latent = scale_intrinsics(target_k, target_hw, latent_hw)
    memory = LatentSpatialMemory(
        capture_latents.shape[1],
        device=device,
        feature_dtype=capture_latents.dtype,
        max_points=args.memory_max_points,
        voxel_size=args.memory_voxel_size,
    )
    for index, frame_index in enumerate(capture_indices):
        candidate_depth = downsample_depth_bilinear(
            capture_depth[index],
            latent_hw,
        )
        candidate_valid = depth_validity_mask(
            candidate_depth,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            relative_edge_threshold=args.depth_edge_threshold,
        )
        if len(memory):
            existing = memory.read(
                capture_c2w[index],
                capture_k_latent[index],
                latent_hw,
            )
            candidate_valid = memory_consistency_mask(
                candidate_depth,
                candidate_valid,
                existing.depth[0, 0],
                existing.visibility[0, 0],
                relative_threshold=args.memory_consistency_threshold,
            )
        memory.write(
            capture_latents[index],
            capture_depth[index],
            capture_k[index],
            capture_c2w[index],
            image_hw=target_hw,
            valid_mask=candidate_valid,
            frame_id=frame_index,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            relative_edge_threshold=args.depth_edge_threshold,
            compact=False,
        )
    memory.compact()
    reference_latent = capture_latents[:1].permute(1, 0, 2, 3).unsqueeze(0)
    prompt_embeds, prompt_mask = pipe.encode_prompt(args.prompt, device=device)
    transformer_dtype = next(model.backbone.parameters()).dtype

    capture_latents_per_clip = 1 + (args.capture_clip_rgb_frames - 1) // 4
    if capture_latents_per_clip < 3:
        raise ValueError("the initial chunk requires a capture clip with >=3 latents")
    final_clip_latents = capture_latents[-capture_latents_per_clip:]
    initial_preceding = final_clip_latents[-3:-1].permute(
        1,
        0,
        2,
        3,
    ).unsqueeze(0)
    initial_prefix = final_clip_latents[-1].unsqueeze(0)
    initial_preceding_c2w = capture_c2w[-3:-1]
    initial_preceding_k_latent = capture_k_latent[-3:-1]

    all_latents = []
    all_depths = []
    generated_latents: list[torch.Tensor] = []
    prefix = initial_prefix
    chunk_stride = args.chunk_latent_frames - 1
    chunk_count = (target_latent_count - 1) // chunk_stride
    for chunk_index in range(chunk_count):
        start = chunk_index * chunk_stride
        end = start + args.chunk_latent_frames
        chunk_c2w = target_c2w[start:end]
        chunk_k = target_k_latent[start:end]
        if chunk_index == 0:
            preceding_latents = initial_preceding
            preceding_c2w = initial_preceding_c2w
            preceding_k = initial_preceding_k_latent
        else:
            preceding_start = max(0, start - 2)
            preceding_latents = torch.stack(
                generated_latents[preceding_start:start],
                dim=1,
            ).unsqueeze(0)
            preceding_c2w = target_c2w[preceding_start:start]
            preceding_k = target_k_latent[preceding_start:start]
        condition_c2w = torch.cat((chunk_c2w, preceding_c2w), dim=0)
        condition_k = torch.cat((chunk_k, preceding_k), dim=0)
        readout = memory.read(condition_c2w, condition_k, latent_hw)
        memory_latents = readout.features.unsqueeze(0)
        visibility = readout.visibility.unsqueeze(0)
        projected_depth = readout.depth.unsqueeze(0)
        rays = make_plucker_rays(
            chunk_c2w.unsqueeze(0),
            chunk_k.unsqueeze(0),
            *latent_hw,
        ).transpose(1, 2)
        preceding_rays = make_plucker_rays(
            preceding_c2w.unsqueeze(0),
            preceding_k.unsqueeze(0),
            *latent_hw,
        ).transpose(1, 2)
        latents = torch.randn(
            1,
            capture_latents.shape[1],
            args.chunk_latent_frames,
            *latent_hw,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        latents[:, :, 0] = prefix
        pipe.scheduler.set_timesteps(args.steps, device=device, shift=args.shift)
        for timestep in pipe.scheduler.timesteps:
            timestep_batch = _transformer_timestep(
                timestep,
                transformer_dtype,
            ).expand(1).to(device)
            target_timesteps = timestep_batch[:, None].expand(
                1,
                args.chunk_latent_frames,
            ).clone()
            target_timesteps[:, 0] = 0
            with torch.no_grad(), _transformer_autocast(device, transformer_dtype):
                velocity = model(
                    noisy_latents=latents,
                    target_timesteps=target_timesteps,
                    encoder_hidden_states=prompt_embeds.to(transformer_dtype),
                    encoder_attention_mask=prompt_mask,
                    memory_latents=memory_latents,
                    memory_visibility=visibility,
                    target_rays=rays,
                    preceding_latents=preceding_latents,
                    preceding_rays=preceding_rays,
                    reference_latents=reference_latent,
                ).velocity.float()
            latents = pipe.scheduler.step(
                velocity,
                timestep,
                latents,
                return_dict=False,
                generator=generator,
            )[0]
            latents[:, :, 0] = prefix
        with torch.no_grad(), _transformer_autocast(device, transformer_dtype):
            predicted_depth = model.predict_log_depth(
                latents,
                rays,
                projected_depth[:, :, : args.chunk_latent_frames],
                visibility[:, :, : args.chunk_latent_frames],
            ).exp().squeeze(1)
        valid = (
            torch.isfinite(predicted_depth[0, 1:])
            & (predicted_depth[0, 1:] >= args.min_depth)
            & (predicted_depth[0, 1:] <= args.max_depth)
        )
        valid = memory_consistency_mask(
            predicted_depth[0, 1:],
            valid,
            projected_depth[0, 0, 1 : args.chunk_latent_frames],
            visibility[0, 0, 1 : args.chunk_latent_frames],
            relative_threshold=args.memory_consistency_threshold,
        )
        memory.write_video(
            latents[0, :, 1:].to(capture_latents.dtype),
            predicted_depth[0, 1:],
            chunk_k[1:],
            chunk_c2w[1:],
            image_hw=latent_hw,
            valid_masks=valid,
            frame_ids=torch.tensor(target_indices[start + 1 : end], device=device),
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            relative_edge_threshold=args.depth_edge_threshold,
        )
        all_latents.append(latents.cpu() if chunk_index == 0 else latents[:, :, 1:].cpu())
        all_depths.append(
            predicted_depth.cpu()
            if chunk_index == 0
            else predicted_depth[:, 1:].cpu()
        )
        new_generated = latents[0] if chunk_index == 0 else latents[0, :, 1:]
        generated_latents.extend(
            list(torch.unbind(new_generated.detach(), dim=1))
        )
        prefix = latents[:, :, -1].detach()
        print(
            f"chunk {chunk_index + 1}/{chunk_count}: "
            f"memory_points={len(memory)} visible={float(visibility.mean()):.4f}",
            flush=True,
        )

    latent_video = torch.cat(all_latents, dim=2).to(device)
    with torch.no_grad():
        frames = pipe._decode_latents(latent_video)[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output, (frames * 255.0).round().astype(np.uint8), fps=args.fps)
    np.savez_compressed(
        output.with_suffix(".depth.npz"),
        depth=torch.cat(all_depths, dim=1).numpy(),
        latent_frame_indices=np.asarray(target_indices),
        normalized_c2w=target_c2w.cpu().numpy(),
        intrinsics=target_k.cpu().numpy(),
    )
    del model, pipe
    gc.collect()


if __name__ == "__main__":
    main()
