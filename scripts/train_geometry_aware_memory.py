from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.geometry_aware_memory.data import (  # noqa: E402
    GeometryMemorySampleConfig,
    LocalVipeRoomTourDataset,
    RoomTourSample,
)
from lingbot_video.geometry_aware_memory.model import (  # noqa: E402
    GIMWorldLingBotModel,
    GIMWorldModelConfig,
)
from lingbot_video.geometry_aware_memory.pruning import (  # noqa: E402
    MIGreedyPruner,
    PoseTimeKernelConfig,
)
from lingbot_video.geometry_aware_memory.teacher import (  # noqa: E402
    VGGTGeometryTeacher,
    VGGTTeacherConfig,
    vggt_target_hw,
)
from lingbot_video.geometry_aware_memory.training import (  # noqa: E402
    GIMTrainingConfig,
    SceneLatentCache,
    gim_training_step,
    prepare_gim_batch,
)
from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline  # noqa: E402
from lingbot_video.runner import _patch_qwen3vl_from_pretrained  # noqa: E402
from lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModel,
)


logger = logging.getLogger("lingbot_video.train_geometry_aware_memory")


def _config_defaults() -> dict[str, Any]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    known, _ = bootstrap.parse_known_args()
    if not known.config:
        return {}
    payload = json.loads(Path(known.config).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("training config must be one JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GIM-World geometry-aware memory on LingBot-Video."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_dir", required=False)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument(
        "--latent_cache_root",
        default=None,
    )
    parser.add_argument(
        "--teacher_cache_root",
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/gim_world_geometry_memory",
    )
    parser.add_argument("--prompt", default="An indoor room tour.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_rgb_frames", type=int, default=81)
    parser.add_argument("--vae_temporal_stride", type=int, default=4)
    parser.add_argument("--vae_encode_chunk_rgb_frames", type=int, default=81)
    parser.add_argument("--min_memory_rgb_frames", type=int, default=800)
    parser.add_argument("--target_guard_rgb_frames", type=int, default=128)
    parser.add_argument(
        "--context_policy",
        choices=["all_except_target", "prefix"],
        default="prefix",
    )
    parser.add_argument("--samples_per_item", type=int, default=16)
    parser.add_argument("--pruning_budget", type=int, default=200)
    parser.add_argument("--memory_latent_frames", type=int, default=20)
    parser.add_argument("--compact_stride", type=int, default=2)
    parser.add_argument("--geometry_loss_weight", type=float, default=0.05)
    parser.add_argument("--sigma_position", type=float, default=-1.0)
    parser.add_argument("--sigma_rotation", type=float, default=math.pi / 6.0)
    parser.add_argument("--sigma_time", type=float, default=50.0)
    parser.add_argument("--kernel_jitter", type=float, default=1e-5)
    parser.add_argument("--timestep_shift", type=float, default=1.0)
    parser.add_argument("--vggt_model_id", default="facebook/VGGT-1B")
    parser.add_argument(
        "--backbone_train_mode",
        choices=["full", "frozen"],
        default="full",
        help="The paper setting is full. frozen is an explicit low-memory ablation.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_train_steps", type=int, default=8000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dataloader_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--item_list", default=None)
    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("--model_dir is required (it may be supplied by --config)")
    if not args.dataset_root:
        parser.error("--dataset_root is required (it may be supplied by --config)")
    if not args.latent_cache_root:
        parser.error(
            "--latent_cache_root is required (it may be supplied by --config)"
        )
    if not args.teacher_cache_root:
        parser.error(
            "--teacher_cache_root is required (it may be supplied by --config)"
        )
    if args.height % 16 or args.width % 16:
        parser.error("--height and --width must be multiples of 16")
    if args.pruning_budget < 1:
        parser.error("--pruning_budget must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader_workers cannot be negative")
    return args


def _dtype(name: str) -> torch.dtype:
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _read_item_list(value: str | list[str] | None) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, list):
        if not all(isinstance(item, str) and item.strip() for item in value):
            raise ValueError("item_list entries must be non-empty strings")
        return [item.strip().rstrip("/") for item in value]
    return [
        line.strip().rstrip("/")
        for line in Path(value).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _first_sample(values: list[RoomTourSample]) -> RoomTourSample:
    if len(values) != 1:
        raise ValueError("GIM long-memory loader currently requires batch size 1")
    return values[0]


def _load_base(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    LingBotVideoTransformer3DModel,
    torch.nn.Module,
    torch.Tensor,
    torch.Tensor,
]:
    compute_dtype = _dtype(args.mixed_precision)
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        args.model_dir,
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
            args.model_dir,
            transformer=transformer,
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
        prompt_embeds, prompt_mask = pipe.encode_prompt(
            args.prompt,
            device=device,
        )
    prompt_embeds = prompt_embeds.detach()
    prompt_mask = prompt_mask.detach()
    vae = pipe.vae.requires_grad_(False).eval().to(device)
    transformer = pipe.transformer
    pipe.text_encoder.to("cpu")
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transformer, vae, prompt_embeds, prompt_mask


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _checkpoint_payload(
    model: GIMWorldLingBotModel,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    step: int,
) -> dict[str, Any]:
    trainable_names = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
        or any(name.startswith(f"{prefix}.") for prefix in (
            "memory_encoder",
            "geometry_head",
            "action_encoder",
        ))
    }
    return {
        "model": state,
        "global_step": int(step),
        "model_config": model.gim_config.to_dict(),
        "training_config": vars(args),
        "optimizer": optimizer.state_dict(),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    torch.manual_seed(args.seed + accelerator.process_index)
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    sample_config = GeometryMemorySampleConfig(
        height=args.height,
        width=args.width,
        target_rgb_frames=args.target_rgb_frames,
        vae_temporal_stride=args.vae_temporal_stride,
        min_memory_rgb_frames=args.min_memory_rgb_frames,
        target_guard_rgb_frames=args.target_guard_rgb_frames,
        samples_per_item=args.samples_per_item,
        context_policy=args.context_policy,
    )
    sample_config.validate()
    dataset = LocalVipeRoomTourDataset(
        args.dataset_root,
        sample_config,
        item_list=_read_item_list(args.item_list),
        seed=args.seed,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.dataloader_workers,
        collate_fn=_first_sample,
        pin_memory=False,
    )
    backbone, vae, prompt_embeds, prompt_mask = _load_base(
        args,
        accelerator.device,
    )
    if args.backbone_train_mode == "frozen":
        backbone.requires_grad_(False)
    else:
        backbone.requires_grad_(True)
    teacher_hw = vggt_target_hw((args.height, args.width))
    model = GIMWorldLingBotModel(
        backbone,
        GIMWorldModelConfig(
            image_height=args.height,
            image_width=args.width,
            memory_latent_frames=args.memory_latent_frames,
            compact_stride=args.compact_stride,
            teacher_grid_height=teacher_hw[0] // 14,
            teacher_grid_width=teacher_hw[1] // 14,
        ),
    )
    if args.gradient_checkpointing:
        model.backbone.enable_gradient_checkpointing()
        model.memory_encoder.gradient_checkpointing = True

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    # The iterable dataset already shards local items by process/worker.  Do
    # not wrap it in Accelerate's IterableDatasetShard a second time.
    model, optimizer = accelerator.prepare(
        model,
        optimizer,
    )
    prompt_embeds = prompt_embeds.to(accelerator.device)
    prompt_mask = prompt_mask.to(accelerator.device)
    teacher = VGGTGeometryTeacher(
        VGGTTeacherConfig(
            model_id=args.vggt_model_id,
            cache_dir=args.teacher_cache_root,
        ),
        device=accelerator.device,
        dtype=_dtype(args.mixed_precision),
    )
    latent_cache = SceneLatentCache(
        args.latent_cache_root,
        height=args.height,
        width=args.width,
        temporal_stride=args.vae_temporal_stride,
        chunk_rgb_frames=args.vae_encode_chunk_rgb_frames,
    )
    pruner = MIGreedyPruner(
        PoseTimeKernelConfig(
            sigma_position=args.sigma_position,
            sigma_rotation=args.sigma_rotation,
            sigma_time=args.sigma_time,
            jitter=args.kernel_jitter,
        )
    )
    training_config = GIMTrainingConfig(
        pruning_budget=args.pruning_budget,
        geometry_loss_weight=args.geometry_loss_weight,
        vae_temporal_stride=args.vae_temporal_stride,
        vae_encode_chunk_rgb_frames=args.vae_encode_chunk_rgb_frames,
        timestep_shift=args.timestep_shift,
    )
    unwrapped = accelerator.unwrap_model(model)
    total_parameters, trainable_parameters = _count_parameters(unwrapped)
    memory_parameters = sum(
        parameter.numel()
        for parameter in unwrapped.memory_encoder.parameters()
    )
    geometry_parameters = sum(
        parameter.numel()
        for parameter in unwrapped.geometry_head.parameters()
    )
    action_parameters = sum(
        parameter.numel()
        for parameter in unwrapped.action_encoder.parameters()
    )
    backbone_total, backbone_trainable = _count_parameters(
        unwrapped.backbone
    )
    patch_height, patch_width = unwrapped.patch_grid
    compact_height = patch_height // args.compact_stride
    compact_width = patch_width // args.compact_stride
    run_summary = {
        "dataset_root": args.dataset_root,
        "data_access": "direct_local_filesystem",
        "dataset_items": len(dataset.items),
        "samples_per_item_reuse": args.samples_per_item,
        "training_samples_per_dataset_pass": (
            len(dataset.items) * args.samples_per_item
        ),
        "resolution": [args.height, args.width],
        "target_rgb_frames": args.target_rgb_frames,
        "target_latent_frames": sample_config.target_latent_frames,
        "context_policy": args.context_policy,
        "target_guard_rgb_frames": args.target_guard_rgb_frames,
        "vae_cache_mode": "wan_official_feature_cache_v1",
        "vae_read_chunk_rgb_frames": args.vae_encode_chunk_rgb_frames,
        "pruning_budget": args.pruning_budget,
        "memory_latent_frames": args.memory_latent_frames,
        "memory_patch_grid": [patch_height, patch_width],
        "memory_token_count": (
            args.memory_latent_frames * patch_height * patch_width
        ),
        "memory_attention_tokens_at_full_budget": (
            (args.memory_latent_frames + args.pruning_budget)
            * compact_height
            * compact_width
        ),
        "compact_stride": args.compact_stride,
        "geometry_loss_weight": args.geometry_loss_weight,
        "backbone_train_mode": args.backbone_train_mode,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "memory_encoder_parameters": memory_parameters,
        "geometry_head_parameters": geometry_parameters,
        "action_encoder_parameters": action_parameters,
        "backbone_parameters": backbone_total,
        "backbone_trainable_parameters": backbone_trainable,
        "world_size": accelerator.num_processes,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": (
            accelerator.num_processes * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "max_train_steps": args.max_train_steps,
    }
    if accelerator.is_main_process:
        logger.info("GIM-WORLD TRAINING CONFIG\n%s", json.dumps(
            run_summary,
            indent=2,
            ensure_ascii=False,
        ))
        (output_dir / "resolved_training_config.json").write_text(
            json.dumps(
                {**vars(args), **run_summary},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.init_trackers(
        "gim_world_geometry_memory",
        config={
            key: value
            for key, value in run_summary.items()
            if isinstance(value, (str, int, float, bool))
        },
    )

    global_step = 0
    if args.resume_from_checkpoint:
        checkpoint = torch.load(
            args.resume_from_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        missing, unexpected = unwrapped.load_state_dict(
            checkpoint["model"],
            strict=False,
        )
        global_step = int(checkpoint.get("global_step", 0))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(
            "resumed components at step=%d missing=%d unexpected=%d",
            global_step,
            len(missing),
            len(unexpected),
        )

    model.train()
    for sample in dataloader:
        if global_step >= args.max_train_steps:
            break
        prepared = prepare_gim_batch(
            sample,
            vae=vae,
            latent_cache=latent_cache,
            pruner=pruner,
            pruning_budget=args.pruning_budget,
            vae_temporal_stride=args.vae_temporal_stride,
            device=accelerator.device,
            compute_dtype=_dtype(args.mixed_precision),
        )
        with accelerator.accumulate(model):
            output = gim_training_step(
                model,
                prepared,
                teacher=teacher,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                config=training_config,
                item_name=sample.item_name,
                geometry_query_index=sample.geometry_query_index,
            )
            accelerator.backward(output["loss"])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(),
                    args.max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if not accelerator.sync_gradients:
            continue
        global_step += 1
        metrics = {
            f"train/{key}": float(value.detach().float().item())
            for key, value in output.items()
        }
        metrics["train/learning_rate"] = float(
            optimizer.param_groups[0]["lr"]
        )
        accelerator.log(metrics, step=global_step)
        if global_step % args.log_every == 0 and accelerator.is_main_process:
            logger.info(
                "step=%d item=%s target_start=%d %s",
                global_step,
                sample.item_name,
                sample.target_start,
                " ".join(
                    f"{key}={value:.6g}"
                    for key, value in metrics.items()
                ),
            )
        if (
            global_step % args.checkpoint_every == 0
            or global_step == args.max_train_steps
        ):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_dir = output_dir / f"checkpoint-{global_step:08d}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                payload = _checkpoint_payload(
                    accelerator.unwrap_model(model),
                    optimizer,
                    args=args,
                    step=global_step,
                )
                accelerator.save(
                    payload,
                    checkpoint_dir / "trainable_components.pt",
                )
                logger.info("saved %s", checkpoint_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
