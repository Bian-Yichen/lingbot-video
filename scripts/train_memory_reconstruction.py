from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from diffusers import AutoencoderKLWan
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.geometry_aware_memory.data import (  # noqa: E402
    GeometryMemorySampleConfig,
    LocalVipeRoomTourDataset,
    RoomTourSample,
)
from lingbot_video.geometry_aware_memory.memory_reconstruction import (  # noqa: E402
    GIMMemoryReconstructionModel,
    load_lingbot_patch_embedder,
    reconstruction_config_from_lingbot,
)
from lingbot_video.geometry_aware_memory.reconstruction_training import (  # noqa: E402
    ReconstructionObjective,
    prepare_reconstruction_batch,
    reconstruction_training_step,
)


logger = logging.getLogger("lingbot_video.train_memory_reconstruction")


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
            "Train the existing GIM memory encoder through held-out novel-view "
            "RGB reconstruction, without loading or executing the LingBot DiT."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument(
        "--output_dir",
        default="outputs/gim_memory_novel_view_decoder",
    )
    parser.add_argument("--item_list", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)

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

    parser.add_argument("--memory_latent_frames", type=int, default=1)
    parser.add_argument("--compact_stride", type=int, default=2)
    parser.add_argument("--decoder_hidden_size", type=int, default=768)
    parser.add_argument("--decoder_depth", type=int, default=16)
    parser.add_argument("--decoder_num_heads", type=int, default=16)
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument(
        "--train_patch_embedder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lpips_weight", type=float, default=1.0)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=8000)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
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
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every_iterations", type=int, default=200)
    parser.add_argument(
        "--save_optimizer_state",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    for name in ("model_dir", "dataset_root", "output_dir"):
        if not getattr(args, name):
            parser.error(f"--{name} is required (it may be supplied by --config)")
    positive = (
        "target_rgb_frames",
        "query_blocks",
        "local_window_rgb_frames",
        "memory_views_min",
        "memory_views_max",
        "vae_encode_chunk_rgb_frames",
        "memory_latent_frames",
        "compact_stride",
        "decoder_hidden_size",
        "decoder_depth",
        "decoder_num_heads",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "log_every",
        "checkpoint_every_iterations",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.lpips_weight < 0:
        parser.error("--lpips_weight cannot be negative")
    if args.lr_warmup_steps < 0:
        parser.error("--lr_warmup_steps cannot be negative")
    if args.decoder_hidden_size % args.decoder_num_heads:
        parser.error("decoder_hidden_size must be divisible by decoder_num_heads")
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


def _checkpoint_file(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "memory_reconstruction.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _checkpoint_payload(
    model: GIMMemoryReconstructionModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    args: argparse.Namespace,
    global_step: int,
    global_iteration: int,
    next_epoch: int,
    next_iteration_in_epoch: int,
    epoch_loader_generator_state: torch.Tensor | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "model_config": model.reconstruction_config.to_dict(),
        "training_config": vars(args),
        "global_step": int(global_step),
        "global_iteration": int(global_iteration),
        "next_epoch": int(next_epoch),
        "next_iteration_in_epoch": int(next_iteration_in_epoch),
    }
    if epoch_loader_generator_state is not None:
        payload["epoch_loader_generator_state"] = (
            epoch_loader_generator_state.cpu()
        )
    if args.save_optimizer_state:
        payload["optimizer"] = optimizer.state_dict()
    payload["scheduler"] = scheduler.state_dict()
    return payload


def _save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    *,
    args: argparse.Namespace,
    global_step: int,
    global_iteration: int,
    next_epoch: int,
    next_iteration_in_epoch: int,
    epoch_loader_generator_state: torch.Tensor | None,
) -> Path:
    checkpoint_dir = output_dir / (
        f"checkpoint-iter-{global_iteration:08d}-step-{global_step:08d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save(
        _checkpoint_payload(
            accelerator.unwrap_model(model),
            optimizer,
            scheduler,
            args=args,
            global_step=global_step,
            global_iteration=global_iteration,
            next_epoch=next_epoch,
            next_iteration_in_epoch=next_iteration_in_epoch,
            epoch_loader_generator_state=epoch_loader_generator_state,
        ),
        checkpoint_dir / "memory_reconstruction.pt",
    )
    return checkpoint_dir


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def _cosine_schedule(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps < 1:
        raise ValueError("total optimizer steps must be positive")

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _load_vae(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    vae = AutoencoderKLWan.from_pretrained(
        args.model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    return vae.requires_grad_(False).eval().to(device)


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
        local_window_rgb_frames=args.local_window_rgb_frames,
        memory_views_min=args.memory_views_min,
        memory_views_max=args.memory_views_max,
        retrieval_rotation_weight=args.retrieval_rotation_weight,
        retrieval_temperature=args.retrieval_temperature,
    )
    sample_config.validate()
    dataset = LocalVipeRoomTourDataset(
        args.dataset_root,
        sample_config,
        item_list=_read_item_list(args.item_list),
        seed=args.seed,
        preload_training_inputs=True,
    )
    dataloader_kwargs: dict[str, Any] = {}
    if args.dataloader_workers > 0:
        dataloader_kwargs["prefetch_factor"] = args.dataloader_prefetch_factor
    loader_generator = torch.Generator().manual_seed(args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.dataloader_workers,
        collate_fn=_first_sample,
        pin_memory=False,
        persistent_workers=False,
        generator=loader_generator,
        **dataloader_kwargs,
    )

    patch_config, patch_embedder = load_lingbot_patch_embedder(args.model_dir)
    reconstruction_config = reconstruction_config_from_lingbot(
        patch_config,
        image_height=args.height,
        image_width=args.width,
        memory_latent_frames=args.memory_latent_frames,
        compact_stride=args.compact_stride,
        decoder_hidden_size=args.decoder_hidden_size,
        decoder_depth=args.decoder_depth,
        decoder_num_heads=args.decoder_num_heads,
        decoder_mlp_ratio=args.decoder_mlp_ratio,
        train_patch_embedder=args.train_patch_embedder,
    )
    model = GIMMemoryReconstructionModel(
        patch_embedder,
        reconstruction_config,
    )
    if args.gradient_checkpointing:
        model.memory_encoder.gradient_checkpointing = True
        model.decoder.gradient_checkpointing = True
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    model, optimizer, dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader,
    )
    total_optimizer_steps = args.num_train_epochs * math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    scheduler_optimizer = getattr(optimizer, "optimizer", optimizer)
    scheduler = _cosine_schedule(
        scheduler_optimizer,
        warmup_steps=args.lr_warmup_steps,
        total_steps=total_optimizer_steps,
    )
    objective = ReconstructionObjective(args.lpips_weight).to(accelerator.device)
    vae = _load_vae(args, accelerator.device)

    unwrapped = accelerator.unwrap_model(model)
    total_parameters, trainable_parameters = _count_parameters(unwrapped)
    memory_parameters = _count_parameters(unwrapped.memory_encoder)
    decoder_parameters = _count_parameters(unwrapped.decoder)
    patch_grid = unwrapped.patch_grid
    run_summary = {
        "experiment": "no_dit_gim_memory_novel_view_reconstruction",
        "dataset_root": args.dataset_root,
        "dataset_items": len(dataset),
        "dataset_items_total": dataset.total_item_count,
        "dataset_items_skipped_epoch_1": dataset.skipped_item_count,
        "resolution": [args.height, args.width],
        "history_views": [args.memory_views_min, args.memory_views_max],
        "target_views_per_scene": sample_config.query_rgb_frames,
        "history_target_overlap": 0,
        "target_supervision": "held_out_novel_view_rgb",
        "history_feature_extractor": "frozen_wan_vae_and_lingbot_patch_projection",
        "dit_loaded": False,
        "dit_executed": False,
        "text_encoder_loaded": False,
        "memory_patch_grid": list(patch_grid),
        "memory_tokens": args.memory_latent_frames * patch_grid[0] * patch_grid[1],
        "memory_encoder_architecture": "existing_gim_two_block_compact_encoder",
        "decoder_architecture": "plucker_ray_latent_query_transformer",
        "decoder_depth": args.decoder_depth,
        "decoder_hidden_size": args.decoder_hidden_size,
        "decoder_num_heads": args.decoder_num_heads,
        "loss": "rgb_mse_plus_lpips",
        "lpips_weight": args.lpips_weight,
        "gan_loss": "disabled_paper_ablation",
        "point_map_loss": "disabled_no_geometry_supervision",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "memory_encoder_parameters": memory_parameters[0],
        "memory_encoder_trainable_parameters": memory_parameters[1],
        "decoder_parameters": decoder_parameters[0],
        "decoder_trainable_parameters": decoder_parameters[1],
        "learning_rate": args.learning_rate,
        "lr_schedule": "linear_warmup_then_cosine_decay",
        "lr_warmup_steps": args.lr_warmup_steps,
        "total_optimizer_steps": total_optimizer_steps,
        "world_size": accelerator.num_processes,
    }
    if accelerator.is_main_process:
        logger.info(
            "MEMORY RECONSTRUCTION TRAINING CONFIG\n%s",
            json.dumps(run_summary, indent=2, ensure_ascii=False),
        )
        (output_dir / "resolved_training_config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    **run_summary,
                    "model_config": reconstruction_config.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.init_trackers(
        "gim_memory_novel_view_reconstruction",
        config={
            key: value
            for key, value in run_summary.items()
            if isinstance(value, (str, int, float, bool))
        },
    )

    global_step = 0
    global_iteration = 0
    start_epoch = 0
    resume_iteration_in_epoch = 0
    resume_loader_generator_state: torch.Tensor | None = None
    if args.resume_from_checkpoint:
        checkpoint_payload = torch.load(
            _checkpoint_file(args.resume_from_checkpoint),
            map_location="cpu",
            weights_only=False,
        )
        unwrapped.load_state_dict(checkpoint_payload["model"], strict=True)
        if "optimizer" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer"])
        else:
            logger.warning(
                "checkpoint has no optimizer state; optimizer restarts fresh"
            )
        if "scheduler" in checkpoint_payload:
            scheduler.load_state_dict(checkpoint_payload["scheduler"])
            for parameter_group, learning_rate in zip(
                scheduler_optimizer.param_groups,
                scheduler.get_last_lr(),
                strict=True,
            ):
                parameter_group["lr"] = learning_rate
        global_step = int(checkpoint_payload.get("global_step", 0))
        global_iteration = int(checkpoint_payload.get("global_iteration", 0))
        start_epoch = int(checkpoint_payload.get("next_epoch", 0))
        resume_iteration_in_epoch = int(
            checkpoint_payload.get("next_iteration_in_epoch", 0)
        )
        resume_loader_generator_state = checkpoint_payload.get(
            "epoch_loader_generator_state"
        )

    next_checkpoint = (
        global_iteration // args.checkpoint_every_iterations + 1
    ) * args.checkpoint_every_iterations
    last_checkpoint_iteration = -1
    model.train()
    for epoch in range(start_epoch, args.num_train_epochs):
        dataset.set_epoch(epoch)
        logger.info(
            "epoch=%d/%d scenes=%d history_views=%d..%d novel_targets=%d",
            epoch + 1,
            args.num_train_epochs,
            len(dataset),
            args.memory_views_min,
            args.memory_views_max,
            sample_config.query_rgb_frames,
        )
        iteration_offset = (
            resume_iteration_in_epoch if epoch == start_epoch else 0
        )
        if iteration_offset < 0 or iteration_offset > len(dataloader):
            raise ValueError(
                "checkpoint next_iteration_in_epoch is outside the current "
                f"dataloader: {iteration_offset} vs {len(dataloader)}"
            )
        if epoch == start_epoch and resume_loader_generator_state is not None:
            loader_generator.set_state(resume_loader_generator_state)
        epoch_loader_generator_state = loader_generator.get_state().clone()
        dataloader_iterator = iter(dataloader)
        for _ in range(iteration_offset):
            next(dataloader_iterator)
        for scene_step in range(iteration_offset, len(dataloader)):
            sample = next(dataloader_iterator)
            started_at = time.perf_counter()
            prepared = prepare_reconstruction_batch(
                sample,
                vae=vae,
                vae_encode_chunk_rgb_frames=args.vae_encode_chunk_rgb_frames,
                device=accelerator.device,
                compute_dtype=_dtype(args.mixed_precision),
            )
            with accelerator.accumulate(model):
                output = reconstruction_training_step(
                    model,
                    prepared,
                    objective=objective,
                    device=accelerator.device,
                    compute_dtype=_dtype(args.mixed_precision),
                )
                accelerator.backward(output["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(),
                        args.max_grad_norm,
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_iteration += 1
            if accelerator.sync_gradients:
                global_step += 1
            local_metrics = torch.stack(
                [
                    output["loss"].detach().float(),
                    output["rgb_mse"],
                    output["lpips"],
                    output["psnr"],
                    output["memory_norm"],
                ]
            ).unsqueeze(0)
            gathered = accelerator.gather(local_metrics)
            means = gathered.reshape(-1, 5).mean(dim=0)
            if accelerator.is_main_process:
                logger.info(
                    "scene epoch=%d/%d iter=%d/%d item=%s global_iter=%d "
                    "global_step=%d loss=%.6g mse=%.6g lpips=%.6g psnr=%.3f "
                    "memory_norm=%.5f history=%d target=%d elapsed=%.1fs",
                    epoch + 1,
                    args.num_train_epochs,
                    scene_step + 1,
                    len(dataloader),
                    sample.item_name,
                    global_iteration,
                    global_step,
                    means[0].item(),
                    means[1].item(),
                    means[2].item(),
                    means[3].item(),
                    means[4].item(),
                    len(sample.capture_rgb_indices),
                    sample_config.query_rgb_frames,
                    time.perf_counter() - started_at,
                )
            if accelerator.sync_gradients:
                accelerator.log(
                    {
                        "train/loss": means[0].item(),
                        "train/rgb_mse": means[1].item(),
                        "train/lpips": means[2].item(),
                        "train/psnr": means[3].item(),
                        "train/memory_norm": means[4].item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )
            if (
                global_iteration >= next_checkpoint
                and accelerator.sync_gradients
            ):
                next_iteration_in_epoch = scene_step + 1
                if next_iteration_in_epoch >= len(dataloader):
                    checkpoint_next_epoch = epoch + 1
                    next_iteration_in_epoch = 0
                    checkpoint_loader_state = None
                else:
                    checkpoint_next_epoch = epoch
                    checkpoint_loader_state = epoch_loader_generator_state
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_dir = _save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        scheduler,
                        output_dir,
                        args=args,
                        global_step=global_step,
                        global_iteration=global_iteration,
                        next_epoch=checkpoint_next_epoch,
                        next_iteration_in_epoch=next_iteration_in_epoch,
                        epoch_loader_generator_state=checkpoint_loader_state,
                    )
                    logger.info("saved %s", checkpoint_dir)
                accelerator.wait_for_everyone()
                last_checkpoint_iteration = global_iteration
                while next_checkpoint <= global_iteration:
                    next_checkpoint += args.checkpoint_every_iterations
        resume_iteration_in_epoch = 0
        resume_loader_generator_state = None

    if global_iteration != last_checkpoint_iteration:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_dir = _save_checkpoint(
                accelerator,
                model,
                optimizer,
                scheduler,
                output_dir,
                args=args,
                global_step=global_step,
                global_iteration=global_iteration,
                next_epoch=args.num_train_epochs,
                next_iteration_in_epoch=0,
                epoch_loader_generator_state=None,
            )
            logger.info("saved final %s", checkpoint_dir)
        accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
