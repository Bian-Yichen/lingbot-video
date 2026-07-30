from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import math
import sys
import time
from dataclasses import replace
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
    gim_trajectory_training_step,
    prepare_gim_trajectory_online,
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
        description=(
            "Train GIM-World with interleaved sparse capture/query "
            "room-tour trajectories."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument(
        "--output_dir",
        default="outputs/gim_world_geometry_memory",
    )
    parser.add_argument("--item_list", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--prompt", default="An indoor room tour.")

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_rgb_frames", type=int, default=49)
    parser.add_argument("--query_blocks", type=int, default=1)
    parser.add_argument("--vae_temporal_stride", type=int, default=4)
    parser.add_argument("--vae_encode_chunk_rgb_frames", type=int, default=81)
    parser.add_argument("--capture_window_min_rgb_frames", type=int, default=257)
    parser.add_argument("--capture_window_max_rgb_frames", type=int, default=1000)
    parser.add_argument(
        "--capture_window_curriculum_start_max_rgb_frames",
        type=int,
        default=321,
    )
    parser.add_argument(
        "--capture_window_curriculum_epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--capture_window_min_fraction_of_current_max",
        type=float,
        default=0.75,
    )
    parser.add_argument("--capture_frame_stride_min", type=int, default=2)
    parser.add_argument("--capture_frame_stride_max", type=int, default=3)
    parser.add_argument("--trajectory_pose_stride", type=int, default=4)
    parser.add_argument("--trajectory_rotation_weight", type=float, default=0.25)

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
        "--predicted_update_probability_start",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--predicted_update_probability_end",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--predicted_update_warmup_epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--backbone_train_mode",
        choices=["full", "frozen"],
        default="full",
        help="full is the paper setting; frozen is a memory-only ablation.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=20)
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
    parser.add_argument("--checkpoint_every_epochs", type=int, default=1)

    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    for name in ("model_dir", "dataset_root", "output_dir"):
        if not getattr(args, name):
            parser.error(f"--{name} is required (it may be supplied by --config)")
    if args.num_train_epochs < 1:
        parser.error("--num_train_epochs must be positive")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be positive")
    if args.log_every < 1 or args.checkpoint_every_epochs < 1:
        parser.error("--log_every and --checkpoint_every_epochs must be positive")
    if args.dataloader_workers < 0:
        parser.error("--dataloader_workers cannot be negative")
    if args.pruning_budget < 1:
        parser.error("--pruning_budget must be positive")
    if args.vae_encode_chunk_rgb_frames < 5:
        parser.error("--vae_encode_chunk_rgb_frames must be at least 5")
    if (args.vae_encode_chunk_rgb_frames - 1) % args.vae_temporal_stride:
        parser.error(
            "--vae_encode_chunk_rgb_frames must equal "
            "1 + k * vae_temporal_stride"
        )
    for name in (
        "predicted_update_probability_start",
        "predicted_update_probability_end",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name} must be in [0,1]")
    if args.predicted_update_warmup_epochs < 1:
        parser.error("--predicted_update_warmup_epochs must be positive")
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
        raise ValueError("one dataloader item must contain exactly one scene")
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


def _gather_scene_records(
    record: dict[str, Any],
    accelerator: Accelerator,
) -> list[dict[str, Any]]:
    """Collect one lightweight scene record per rank for ordered logging."""
    if accelerator.num_processes == 1:
        return [record]
    if not torch.distributed.is_available():
        raise RuntimeError(
            "multi-process scene logging requires torch.distributed"
        )
    if not torch.distributed.is_initialized():
        raise RuntimeError(
            "Accelerate reports multiple processes but the distributed "
            "process group is not initialized"
        )
    gathered: list[dict[str, Any] | None] = [
        None for _ in range(accelerator.num_processes)
    ]
    torch.distributed.all_gather_object(gathered, record)
    return [item for item in gathered if item is not None]


def _distributed_mean_metrics(
    output: dict[str, torch.Tensor],
    sample: RoomTourSample,
    *,
    update_probability: float,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
) -> dict[str, float]:
    """Average optimizer-step metrics across all distributed ranks."""
    tensor_metrics = {
        **{
            f"train/{key}": value.detach().float()
            for key, value in output.items()
        },
        "train/capture_rgb_frames": output["loss"].new_tensor(
            len(sample.capture_rgb_indices),
            dtype=torch.float32,
        ),
        "train/capture_window_rgb_frames": output["loss"].new_tensor(
            sample.capture_window_end - sample.capture_window_start + 1,
            dtype=torch.float32,
        ),
        "train/capture_frame_stride": output["loss"].new_tensor(
            sample.capture_frame_stride,
            dtype=torch.float32,
        ),
    }
    names = tuple(tensor_metrics)
    local_values = torch.stack(
        [tensor_metrics[name] for name in names],
        dim=0,
    ).reshape(1, -1)
    gathered_values = accelerator.gather(local_values)
    mean_values = gathered_values.reshape(-1, len(names)).mean(dim=0)
    metrics = {
        name: float(value.item())
        for name, value in zip(names, mean_values, strict=True)
    }
    metrics.update(
        {
            "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train/predicted_update_probability": update_probability,
        }
    )
    return metrics


def _scene_record(
    output: dict[str, torch.Tensor],
    sample: RoomTourSample,
    *,
    accelerator: Accelerator,
    epoch: int,
    num_epochs: int,
    scene_step: int,
    scene_steps: int,
    accumulation_position: int,
    accumulation_group_size: int,
    global_step: int,
    did_optimizer_step: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "rank": accelerator.process_index,
        "epoch": epoch + 1,
        "num_epochs": num_epochs,
        "iteration": scene_step + 1,
        "iterations": scene_steps,
        "item": sample.item_name,
        "global_step": global_step,
        "did_optimizer_step": did_optimizer_step,
        "accumulation_position": accumulation_position,
        "accumulation_group_size": accumulation_group_size,
        "loss": float(output["loss"].detach().float().item()),
        "flow_loss": float(output["flow_loss"].detach().float().item()),
        "geometry_loss": float(
            output["geometry_loss"].detach().float().item()
        ),
        "sigma": float(output["sigma"].detach().float().item()),
        "history_candidates": float(
            output["history_candidates"].detach().float().item()
        ),
        "history_retained": float(
            output["history_retained"].detach().float().item()
        ),
        "capture_latent_frames": float(
            output["capture_latent_frames"].detach().float().item()
        ),
        "memory_norm": float(
            output["memory_norm"].detach().float().item()
        ),
        "capture_window_start": sample.capture_window_start,
        "capture_window_end": sample.capture_window_end,
        "capture_frame_stride": sample.capture_frame_stride,
        "capture_rgb_frames": len(sample.capture_rgb_indices),
        "query_start": sample.query_start,
        "query_end": sample.query_rgb_blocks[-1][-1],
        "overlap_score": sample.trajectory_overlap_score,
        "elapsed_seconds": elapsed_seconds,
    }


def _log_scene_record(record: dict[str, Any]) -> None:
    logger.info(
        "scene epoch=%d/%d iter=%d/%d rank=%d item=%s "
        "global_step=%d optimizer_step=%s accumulation=%d/%d "
        "loss=%.6g flow_loss=%.6g geometry_loss=%.6g sigma=%.4f "
        "window=%d:%d stride=%d capture_rgb=%d capture_latents=%.0f "
        "query=%d:%d overlap=%.4f history=%.0f/%.0f "
        "memory_norm=%.6g elapsed=%.1fs",
        record["epoch"],
        record["num_epochs"],
        record["iteration"],
        record["iterations"],
        record["rank"],
        record["item"],
        record["global_step"],
        "yes" if record["did_optimizer_step"] else "no",
        record["accumulation_position"],
        record["accumulation_group_size"],
        record["loss"],
        record["flow_loss"],
        record["geometry_loss"],
        record["sigma"],
        record["capture_window_start"],
        record["capture_window_end"],
        record["capture_frame_stride"],
        record["capture_rgb_frames"],
        record["capture_latent_frames"],
        record["query_start"],
        record["query_end"],
        record["overlap_score"],
        record["history_retained"],
        record["history_candidates"],
        record["memory_norm"],
        record["elapsed_seconds"],
    )


def _checkpoint_file(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "trainable_components.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _checkpoint_payload(
    model: GIMWorldLingBotModel,
    optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
    step: int,
    next_epoch: int,
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
        or any(
            name.startswith(f"{prefix}.")
            for prefix in (
                "memory_encoder",
                "geometry_head",
                "action_encoder",
            )
        )
    }
    return {
        "model": state,
        "global_step": int(step),
        "next_epoch": int(next_epoch),
        "model_config": model.gim_config.to_dict(),
        "training_config": vars(args),
        "optimizer": optimizer.state_dict(),
    }


def _scheduled_probability(args: argparse.Namespace, epoch: int) -> float:
    if args.predicted_update_warmup_epochs <= 1:
        return float(args.predicted_update_probability_end)
    progress = min(
        max(epoch, 0) / (args.predicted_update_warmup_epochs - 1),
        1.0,
    )
    return float(
        args.predicted_update_probability_start
        + progress
        * (
            args.predicted_update_probability_end
            - args.predicted_update_probability_start
        )
    )


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
        query_blocks=args.query_blocks,
        vae_temporal_stride=args.vae_temporal_stride,
        capture_window_min_rgb_frames=args.capture_window_min_rgb_frames,
        capture_window_max_rgb_frames=args.capture_window_max_rgb_frames,
        capture_window_curriculum_start_max_rgb_frames=(
            args.capture_window_curriculum_start_max_rgb_frames
        ),
        capture_window_curriculum_epochs=(
            args.capture_window_curriculum_epochs
        ),
        capture_window_min_fraction_of_current_max=(
            args.capture_window_min_fraction_of_current_max
        ),
        capture_frame_stride_min=args.capture_frame_stride_min,
        capture_frame_stride_max=args.capture_frame_stride_max,
        trajectory_pose_stride=args.trajectory_pose_stride,
        trajectory_rotation_weight=args.trajectory_rotation_weight,
    )
    sample_config.validate()
    dataset = LocalVipeRoomTourDataset(
        args.dataset_root,
        sample_config,
        item_list=_read_item_list(args.item_list),
        seed=args.seed,
    )
    logger.info(
        "dataset eligibility epoch=1 total=%d eligible=%d skipped=%d%s",
        dataset.total_item_count,
        len(dataset),
        dataset.skipped_item_count,
        (
            " skipped_examples="
            + ",".join(dataset.skipped_items[:5])
            if dataset.skipped_items
            else ""
        ),
    )
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.dataloader_workers,
        collate_fn=_first_sample,
        pin_memory=False,
        generator=loader_generator,
        # Workers are recreated each epoch so they see dataset.set_epoch().
        persistent_workers=False,
    )

    backbone, vae, prompt_embeds, prompt_mask = _load_base(
        args,
        accelerator.device,
    )
    backbone.requires_grad_(args.backbone_train_mode == "full")
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
    model, optimizer, dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader,
    )
    prompt_embeds = prompt_embeds.to(accelerator.device)
    prompt_mask = prompt_mask.to(accelerator.device)

    # No cache object exists in this loop. VGGT is frozen but executes from
    # the sampled target RGB on every query block of every scene iteration.
    teacher = VGGTGeometryTeacher(
        VGGTTeacherConfig(
            model_id=args.vggt_model_id,
        ),
        device=accelerator.device,
        dtype=_dtype(args.mixed_precision),
    )
    pruner = MIGreedyPruner(
        PoseTimeKernelConfig(
            sigma_position=args.sigma_position,
            sigma_rotation=args.sigma_rotation,
            sigma_time=args.sigma_time,
            jitter=args.kernel_jitter,
        )
    )
    base_training_config = GIMTrainingConfig(
        pruning_budget=args.pruning_budget,
        geometry_loss_weight=args.geometry_loss_weight,
        vae_temporal_stride=args.vae_temporal_stride,
        vae_encode_chunk_rgb_frames=args.vae_encode_chunk_rgb_frames,
        timestep_shift=args.timestep_shift,
    )

    unwrapped = accelerator.unwrap_model(model)
    total_parameters, trainable_parameters = _count_parameters(unwrapped)
    backbone_total, backbone_trainable = _count_parameters(unwrapped.backbone)
    patch_height, patch_width = unwrapped.patch_grid
    compact_height = patch_height // args.compact_stride
    compact_width = patch_width // args.compact_stride
    run_summary = {
        "dataset_root": args.dataset_root,
        "data_access": "direct_local_filesystem",
        "dataset_items": len(dataset),
        "dataset_items_total": dataset.total_item_count,
        "dataset_items_skipped_epoch_1": dataset.skipped_item_count,
        "scene_iterations_per_epoch": len(dataset),
        "scene_sampling": (
            "one_random_interleaved_capture_query_window_per_scene_per_epoch"
        ),
        "resolution": [args.height, args.width],
        "capture_window_rgb_frames": [
            args.capture_window_min_rgb_frames,
            args.capture_window_max_rgb_frames,
        ],
        "capture_window_curriculum_start_max": (
            args.capture_window_curriculum_start_max_rgb_frames
        ),
        "capture_window_curriculum_epochs": (
            args.capture_window_curriculum_epochs
        ),
        "capture_window_min_fraction_of_current_max": (
            args.capture_window_min_fraction_of_current_max
        ),
        "capture_frame_stride": [
            args.capture_frame_stride_min,
            args.capture_frame_stride_max,
        ],
        "query_blocks": args.query_blocks,
        "query_rgb_frames_per_block": args.target_rgb_frames,
        "query_rgb_frames_total": sample_config.query_rgb_frames,
        "query_latent_frames_per_block": sample_config.target_latent_frames,
        "capture_query_relation": (
            "different regular-sampling phases inside one shared window"
        ),
        "vae_execution": "online_from_rgb_every_scene_iteration",
        "vggt_execution": "online_from_rgb_every_query_block",
        "persistent_feature_cache": False,
        "vae_read_chunk_rgb_frames": args.vae_encode_chunk_rgb_frames,
        "dynamic_memory_update": (
            "disabled_for_one_block_training"
            if args.query_blocks == 1
            else "between_blocks_only_then_reencode_memory"
        ),
        "predicted_update_probability": [
            args.predicted_update_probability_start,
            args.predicted_update_probability_end,
        ],
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
        "memory_encoder_parameters": sum(
            p.numel() for p in unwrapped.memory_encoder.parameters()
        ),
        "geometry_head_parameters": sum(
            p.numel() for p in unwrapped.geometry_head.parameters()
        ),
        "action_encoder_parameters": sum(
            p.numel() for p in unwrapped.action_encoder.parameters()
        ),
        "backbone_parameters": backbone_total,
        "backbone_trainable_parameters": backbone_trainable,
        "world_size": accelerator.num_processes,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_scene_batch_size": (
            accelerator.num_processes * args.gradient_accumulation_steps
        ),
        "terminal_scene_logging": (
            "every_iteration_all_ranks_gathered_to_main_process"
        ),
        "tensorboard_metric_reduction": "mean_across_distributed_ranks",
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
    }
    if accelerator.is_main_process:
        logger.info(
            "GIM-WORLD TRAINING CONFIG\n%s",
            json.dumps(run_summary, indent=2, ensure_ascii=False),
        )
        (output_dir / "resolved_training_config.json").write_text(
            json.dumps(
                {**vars(args), **run_summary},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        if accelerator.num_processes > 1:
            logger.warning(
                "Distributed dataloader may pad at most world_size-1 scenes "
                "when the dataset size is not divisible by world size. "
                "Single-GPU training visits every scene exactly once/epoch."
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
    start_epoch = 0
    if args.resume_from_checkpoint:
        checkpoint = torch.load(
            _checkpoint_file(args.resume_from_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        missing, unexpected = unwrapped.load_state_dict(
            checkpoint["model"],
            strict=False,
        )
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("next_epoch", 0))
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(
            "resumed epoch=%d step=%d missing=%d unexpected=%d",
            start_epoch,
            global_step,
            len(missing),
            len(unexpected),
        )

    model.train()
    for epoch in range(start_epoch, args.num_train_epochs):
        dataset.set_epoch(epoch)
        update_probability = (
            0.0
            if args.query_blocks == 1
            else _scheduled_probability(args, epoch)
        )
        training_config = replace(
            base_training_config,
            predicted_update_probability=update_probability,
        )
        if accelerator.is_main_process:
            logger.info(
                "epoch=%d/%d scenes=%d/%d skipped=%d "
                "capture_window_bounds=%s "
                "predicted_update_probability=%.3f",
                epoch + 1,
                args.num_train_epochs,
                len(dataset),
                dataset.total_item_count,
                dataset.skipped_item_count,
                sample_config.capture_window_bounds_for_epoch(epoch),
                update_probability,
            )

        for scene_step, sample in enumerate(dataloader):
            scene_started_at = time.perf_counter()
            # Both capture and every target block are freshly read and encoded.
            # The online VAE work intentionally happens once per scene sample.
            prepared = prepare_gim_trajectory_online(
                sample,
                vae=vae,
                vae_temporal_stride=args.vae_temporal_stride,
                vae_encode_chunk_rgb_frames=(
                    args.vae_encode_chunk_rgb_frames
                ),
                device=accelerator.device,
                compute_dtype=_dtype(args.mixed_precision),
            )
            accumulation_offset = (
                scene_step % args.gradient_accumulation_steps
            )
            accumulation_start = scene_step - accumulation_offset
            accumulation_group_size = min(
                args.gradient_accumulation_steps,
                len(dataloader) - accumulation_start,
            )
            # Accelerator always divides backward losses by the configured
            # accumulation count. Rescale the final partial group so every
            # scene retains equal weight instead of being divided by 32.
            partial_group_rescale = (
                args.gradient_accumulation_steps
                / accumulation_group_size
            )

            def backward_scene(loss: torch.Tensor) -> None:
                accelerator.backward(loss * partial_group_rescale)

            with accelerator.accumulate(model):
                output = gim_trajectory_training_step(
                    model,
                    prepared,
                    teacher=teacher,
                    pruner=pruner,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    config=training_config,
                    device=accelerator.device,
                    compute_dtype=_dtype(args.mixed_precision),
                    backward=backward_scene,
                )
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            did_optimizer_step = bool(accelerator.sync_gradients)
            if did_optimizer_step:
                global_step += 1

            record = _scene_record(
                output,
                sample,
                accelerator=accelerator,
                epoch=epoch,
                num_epochs=args.num_train_epochs,
                scene_step=scene_step,
                scene_steps=len(dataloader),
                accumulation_position=accumulation_offset + 1,
                accumulation_group_size=accumulation_group_size,
                global_step=global_step,
                did_optimizer_step=did_optimizer_step,
                elapsed_seconds=time.perf_counter() - scene_started_at,
            )
            gathered_records = _gather_scene_records(record, accelerator)
            if accelerator.is_main_process:
                for gathered_record in gathered_records:
                    _log_scene_record(gathered_record)

            if not did_optimizer_step:
                continue
            metrics = _distributed_mean_metrics(
                output,
                sample,
                update_probability=update_probability,
                optimizer=optimizer,
                accelerator=accelerator,
            )
            metrics["train/epoch"] = float(epoch)
            accelerator.log(metrics, step=global_step)
            if global_step % args.log_every == 0 and accelerator.is_main_process:
                logger.info(
                    "optimizer step=%d epoch=%d distributed_mean %s",
                    global_step,
                    epoch + 1,
                    " ".join(
                        f"{key}={value:.6g}"
                        for key, value in metrics.items()
                        if key
                        in {
                            "train/loss",
                            "train/flow_loss",
                            "train/geometry_loss",
                            "train/history_retained",
                            "train/predicted_update_fraction",
                        }
                    ),
                )

        accelerator.wait_for_everyone()
        should_save = (
            (epoch + 1) % args.checkpoint_every_epochs == 0
            or epoch + 1 == args.num_train_epochs
        )
        if should_save and accelerator.is_main_process:
            checkpoint_dir = (
                output_dir
                / f"checkpoint-epoch-{epoch + 1:04d}-step-{global_step:08d}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            payload = _checkpoint_payload(
                accelerator.unwrap_model(model),
                optimizer,
                args=args,
                step=global_step,
                next_epoch=epoch + 1,
            )
            accelerator.save(
                payload,
                checkpoint_dir / "trainable_components.pt",
            )
            logger.info("saved %s", checkpoint_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
