from __future__ import annotations

import argparse
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
    load_lingbot_patch_embedder,
)
from lingbot_video.geometry_aware_memory.wan_latent_reconstruction import (  # noqa: E402,E501
    GIMWanLatentReconstructionModel,
    HistoryViewCurriculum,
    WanDecoderBridge,
    wan_reconstruction_config_from_lingbot,
)
from lingbot_video.geometry_aware_memory.wan_latent_training import (  # noqa: E402
    WanLatentReconstructionObjective,
    prepare_wan_latent_batch,
    wan_latent_training_step,
)


logger = logging.getLogger("lingbot_video.train_wan_latent_reconstruction")


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
            "Train GIM memory by rendering camera-aligned Wan latents, then "
            "decode them with the pretrained Wan decoder."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default="outputs/gim_wan_latent_decoder")
    parser.add_argument("--item_list", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--target_rgb_frames", type=int, default=2)
    parser.add_argument("--query_blocks", type=int, default=1)
    parser.add_argument("--local_window_rgb_frames", type=int, default=81)
    parser.add_argument("--history_views_start_min", type=int, default=2)
    parser.add_argument("--history_views_start_max", type=int, default=4)
    parser.add_argument("--history_views_end_min", type=int, default=12)
    parser.add_argument("--history_views_end_max", type=int, default=32)
    parser.add_argument("--history_curriculum_epochs", type=int, default=8)
    parser.add_argument("--retrieval_rotation_weight", type=float, default=0.25)
    parser.add_argument("--retrieval_temperature", type=float, default=0.25)
    parser.add_argument("--vae_encode_chunk_rgb_frames", type=int, default=16)

    parser.add_argument("--memory_latent_frames", type=int, default=1)
    parser.add_argument("--compact_stride", type=int, default=2)
    parser.add_argument("--renderer_hidden_size", type=int, default=768)
    parser.add_argument("--renderer_depth", type=int, default=8)
    parser.add_argument("--renderer_num_heads", type=int, default=16)
    parser.add_argument("--renderer_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--decoder_lora_rank", type=int, default=16)
    parser.add_argument("--decoder_lora_alpha", type=float, default=16.0)
    parser.add_argument("--decoder_refiner_hidden_size", type=int, default=64)
    parser.add_argument(
        "--train_patch_embedder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--rgb_loss_weight", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=0.1)
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=8000)
    parser.add_argument("--decoder_warmup_steps", type=int, default=1000)
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
            parser.error(f"--{name} is required")
    positive = (
        "height",
        "width",
        "target_rgb_frames",
        "query_blocks",
        "local_window_rgb_frames",
        "history_curriculum_epochs",
        "vae_encode_chunk_rgb_frames",
        "memory_latent_frames",
        "compact_stride",
        "renderer_hidden_size",
        "renderer_depth",
        "renderer_num_heads",
        "decoder_lora_rank",
        "decoder_refiner_hidden_size",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "checkpoint_every_iterations",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    for name in (
        "renderer_mlp_ratio",
        "decoder_lora_alpha",
        "learning_rate",
        "decoder_learning_rate",
        "max_grad_norm",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.weight_decay < 0:
        parser.error("--weight_decay cannot be negative")
    if not 1 <= args.stage1_epochs <= args.num_train_epochs:
        parser.error("stage1_epochs must be in [1, num_train_epochs]")
    if args.renderer_hidden_size % args.renderer_num_heads:
        parser.error("renderer hidden size must be divisible by head count")
    for name in (
        "latent_loss_weight",
        "rgb_loss_weight",
        "lpips_weight",
        "lr_warmup_steps",
        "decoder_warmup_steps",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} cannot be negative")
    try:
        curriculum = _history_curriculum(args)
        curriculum.validate()
        end_min, end_max = curriculum.views_for_epoch(
            args.history_curriculum_epochs - 1
        )
        GeometryMemorySampleConfig(
            height=args.height,
            width=args.width,
            target_rgb_frames=args.target_rgb_frames,
            query_blocks=args.query_blocks,
            local_window_rgb_frames=args.local_window_rgb_frames,
            memory_views_min=end_min,
            memory_views_max=end_max,
            retrieval_rotation_weight=args.retrieval_rotation_weight,
            retrieval_temperature=args.retrieval_temperature,
        ).validate()
    except ValueError as error:
        parser.error(str(error))
    return args


def _history_curriculum(args: argparse.Namespace) -> HistoryViewCurriculum:
    return HistoryViewCurriculum(
        start_min=args.history_views_start_min,
        start_max=args.history_views_start_max,
        end_min=args.history_views_end_min,
        end_max=args.history_views_end_max,
        curriculum_epochs=args.history_curriculum_epochs,
    )


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
        path = path / "wan_latent_reconstruction.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _two_stage_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    main_warmup_steps: int,
    stage2_start_step: int,
    decoder_warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def cosine(step: int, start: int, warmup: int, end: int) -> float:
        if step < start:
            return 0.0
        local_step = step - start
        if warmup > 0 and local_step < warmup:
            return float(local_step) / float(max(1, warmup))
        progress = float(local_step - warmup) / float(
            max(1, end - start - warmup)
        )
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    schedules = []
    for group in optimizer.param_groups:
        if group.get("group_role") == "decoder_adapter":
            schedules.append(
                lambda step: cosine(
                    step,
                    stage2_start_step,
                    decoder_warmup_steps,
                    total_steps,
                )
            )
        else:
            schedules.append(
                lambda step: cosine(step, 0, main_warmup_steps, total_steps)
            )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)


def _checkpoint_payload(
    model: GIMWanLatentReconstructionModel,
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
            for name, value in model.experiment_state_dict().items()
        },
        "model_config": model.reconstruction_config.to_dict(),
        "training_config": vars(args),
        "global_step": int(global_step),
        "global_iteration": int(global_iteration),
        "next_epoch": int(next_epoch),
        "next_iteration_in_epoch": int(next_iteration_in_epoch),
        "training_stage": int(model.wan_decoder.training_stage),
        "scheduler": scheduler.state_dict(),
    }
    if epoch_loader_generator_state is not None:
        payload["epoch_loader_generator_state"] = (
            epoch_loader_generator_state.cpu()
        )
    if args.save_optimizer_state:
        payload["optimizer"] = optimizer.state_dict()
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
        checkpoint_dir / "wan_latent_reconstruction.pt",
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


def _load_wan_components(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, WanDecoderBridge, tuple[float, ...], tuple[float, ...]]:
    vae = AutoencoderKLWan.from_pretrained(
        args.model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval().requires_grad_(False)
    if vae.config.patch_size is not None:
        raise NotImplementedError("patchified Wan VAE is not supported")
    if int(vae.config.scale_factor_spatial) != 8:
        raise ValueError("this renderer expects Wan VAE spatial stride 8")
    means = tuple(float(value) for value in vae.config.latents_mean)
    stds = tuple(float(value) for value in vae.config.latents_std)
    post_quant_conv = vae.post_quant_conv
    decoder = vae.decoder
    vae.post_quant_conv = None
    vae.decoder = None
    bridge = WanDecoderBridge(
        post_quant_conv,
        decoder,
        latent_channels=len(means),
        refiner_hidden_size=args.decoder_refiner_hidden_size,
        lora_rank=args.decoder_lora_rank,
        lora_alpha=args.decoder_lora_alpha,
    )
    bridge.gradient_checkpointing = args.gradient_checkpointing
    return vae, bridge, means, stds


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

    curriculum = _history_curriculum(args)
    initial_min, initial_max = curriculum.views_for_epoch(0)
    sample_config = GeometryMemorySampleConfig(
        height=args.height,
        width=args.width,
        target_rgb_frames=args.target_rgb_frames,
        query_blocks=args.query_blocks,
        local_window_rgb_frames=args.local_window_rgb_frames,
        memory_views_min=initial_min,
        memory_views_max=initial_max,
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

    vae_encoder, wan_decoder, means, stds = _load_wan_components(args)
    patch_config, patch_embedder = load_lingbot_patch_embedder(args.model_dir)
    model_config = wan_reconstruction_config_from_lingbot(
        patch_config,
        vae_latents_mean=means,
        vae_latents_std=stds,
        image_height=args.height,
        image_width=args.width,
        memory_latent_frames=args.memory_latent_frames,
        compact_stride=args.compact_stride,
        renderer_hidden_size=args.renderer_hidden_size,
        renderer_depth=args.renderer_depth,
        renderer_num_heads=args.renderer_num_heads,
        renderer_mlp_ratio=args.renderer_mlp_ratio,
        decoder_lora_rank=args.decoder_lora_rank,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_refiner_hidden_size=args.decoder_refiner_hidden_size,
        train_patch_embedder=args.train_patch_embedder,
    )
    model = GIMWanLatentReconstructionModel(
        patch_embedder,
        wan_decoder,
        model_config,
    )
    if args.gradient_checkpointing:
        model.memory_encoder.gradient_checkpointing = True
        model.latent_renderer.gradient_checkpointing = True
    main_parameters = [
        *model.memory_encoder.parameters(),
        *model.latent_renderer.parameters(),
    ]
    if args.train_patch_embedder:
        main_parameters.extend(model.patch_embedder.parameters())
    decoder_parameters = model.wan_decoder.adapter_parameters()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": main_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "group_role": "memory_renderer",
            },
            {
                "params": decoder_parameters,
                "lr": args.decoder_learning_rate,
                "weight_decay": 0.0,
                "group_role": "decoder_adapter",
            },
        ],
        betas=(0.9, 0.95),
    )
    model, optimizer, dataloader = accelerator.prepare(
        model,
        optimizer,
        dataloader,
    )
    vae_encoder.to(accelerator.device)
    steps_per_epoch = math.ceil(
        len(dataloader) / args.gradient_accumulation_steps
    )
    total_optimizer_steps = args.num_train_epochs * steps_per_epoch
    stage2_start_step = args.stage1_epochs * steps_per_epoch
    scheduler_optimizer = getattr(optimizer, "optimizer", optimizer)
    scheduler = _two_stage_scheduler(
        scheduler_optimizer,
        total_steps=total_optimizer_steps,
        main_warmup_steps=args.lr_warmup_steps,
        stage2_start_step=stage2_start_step,
        decoder_warmup_steps=args.decoder_warmup_steps,
    )
    objective = WanLatentReconstructionObjective(
        latent_weight=args.latent_loss_weight,
        rgb_weight=args.rgb_loss_weight,
        lpips_weight=args.lpips_weight,
    ).to(accelerator.device)

    unwrapped = accelerator.unwrap_model(model)
    total_parameters, trainable_parameters = _count_parameters(unwrapped)
    run_summary = {
        "experiment": "gim_memory_to_camera_aligned_wan_latent",
        "dataset_items": len(dataset),
        "resolution": [args.height, args.width],
        "history_curriculum_start": [initial_min, initial_max],
        "history_curriculum_end": [
            args.history_views_end_min,
            args.history_views_end_max,
        ],
        "history_curriculum_epochs": args.history_curriculum_epochs,
        "target_views_per_scene": sample_config.query_rgb_frames,
        "history_target_overlap": 0,
        "stage_1": "train_memory_and_renderer_frozen_wan_decoder",
        "stage_1_epochs": args.stage1_epochs,
        "stage_2": "continue_memory_renderer_enable_wan_decoder_adapters",
        "stage_2_epochs": args.num_train_epochs - args.stage1_epochs,
        "wan_decoder_base_trainable": False,
        "wan_decoder_gradient_checkpointing": (
            unwrapped.wan_decoder.gradient_checkpointing
        ),
        "wan_decoder_adapter_parameters": (
            unwrapped.wan_decoder.adapter_parameter_count
        ),
        "wan_decoder_lora_modules": list(
            unwrapped.wan_decoder.lora_module_names
        ),
        "loss": "normalized_latent_mse_plus_rgb_mse_plus_lpips",
        "latent_loss_weight": args.latent_loss_weight,
        "rgb_loss_weight": args.rgb_loss_weight,
        "lpips_weight": args.lpips_weight,
        "memory_tokens": (
            args.memory_latent_frames
            * unwrapped.patch_grid[0]
            * unwrapped.patch_grid[1]
        ),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "total_optimizer_steps": total_optimizer_steps,
        "stage2_start_step": stage2_start_step,
    }
    if accelerator.is_main_process:
        logger.info(
            "WAN LATENT RECONSTRUCTION CONFIG\n%s",
            json.dumps(run_summary, indent=2, ensure_ascii=False),
        )
        (output_dir / "resolved_training_config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    **run_summary,
                    "model_config": model_config.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.init_trackers(
        "gim_wan_latent_reconstruction",
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
        unwrapped.load_experiment_state_dict(checkpoint_payload["model"])
        if "optimizer" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer"])
        else:
            logger.warning("checkpoint has no optimizer state; optimizer restarts")
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
        stage = 1 if epoch < args.stage1_epochs else 2
        unwrapped.set_training_stage(stage)
        history_min, history_max = curriculum.views_for_epoch(epoch)
        epoch_sample_config = replace(
            sample_config,
            memory_views_min=history_min,
            memory_views_max=history_max,
        )
        epoch_sample_config.validate()
        dataset.sample_config = epoch_sample_config
        dataset.set_epoch(epoch)
        logger.info(
            "epoch=%d/%d stage=%d scenes=%d history_views=%d..%d target=%d",
            epoch + 1,
            args.num_train_epochs,
            stage,
            len(dataset),
            history_min,
            history_max,
            sample_config.query_rgb_frames,
        )
        iteration_offset = (
            resume_iteration_in_epoch if epoch == start_epoch else 0
        )
        if iteration_offset < 0 or iteration_offset > len(dataloader):
            raise ValueError("checkpoint iteration offset is outside dataloader")
        if epoch == start_epoch and resume_loader_generator_state is not None:
            loader_generator.set_state(resume_loader_generator_state)
        epoch_loader_generator_state = loader_generator.get_state().clone()
        dataloader_iterator = iter(dataloader)
        for _ in range(iteration_offset):
            next(dataloader_iterator)
        for scene_step in range(iteration_offset, len(dataloader)):
            sample = next(dataloader_iterator)
            started_at = time.perf_counter()
            prepared = prepare_wan_latent_batch(
                sample,
                vae=vae_encoder,
                vae_encode_chunk_rgb_frames=args.vae_encode_chunk_rgb_frames,
                device=accelerator.device,
                compute_dtype=_dtype(args.mixed_precision),
            )
            with accelerator.accumulate(model):
                output = wan_latent_training_step(
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
            metric_names = (
                "loss",
                "latent_mse",
                "rgb_mse",
                "lpips",
                "psnr",
                "memory_norm",
                "latent_prediction_std",
                "rgb_prediction_std",
            )
            local_metrics = torch.stack(
                [output[name].detach().float() for name in metric_names]
            ).unsqueeze(0)
            gathered = accelerator.gather(local_metrics)
            means = gathered.reshape(-1, len(metric_names)).mean(dim=0)
            metrics = {
                name: float(value.item())
                for name, value in zip(metric_names, means, strict=True)
            }
            if accelerator.is_main_process:
                logger.info(
                    "scene epoch=%d/%d stage=%d iter=%d/%d item=%s "
                    "global_step=%d loss=%.6g latent_mse=%.6g rgb_mse=%.6g "
                    "lpips=%.6g psnr=%.3f history=%d elapsed=%.1fs",
                    epoch + 1,
                    args.num_train_epochs,
                    stage,
                    scene_step + 1,
                    len(dataloader),
                    sample.item_name,
                    global_step,
                    metrics["loss"],
                    metrics["latent_mse"],
                    metrics["rgb_mse"],
                    metrics["lpips"],
                    metrics["psnr"],
                    len(sample.capture_rgb_indices),
                    time.perf_counter() - started_at,
                )
            if accelerator.sync_gradients:
                accelerator.log(
                    {
                        **{f"train/{key}": value for key, value in metrics.items()},
                        "train/stage": stage,
                        "train/history_views_min": history_min,
                        "train/history_views_max": history_max,
                        "train/main_lr": optimizer.param_groups[0]["lr"],
                        "train/decoder_lr": optimizer.param_groups[1]["lr"],
                    },
                    step=global_step,
                )
            if (
                global_iteration >= next_checkpoint
                and accelerator.sync_gradients
            ):
                next_iteration = scene_step + 1
                if next_iteration >= len(dataloader):
                    checkpoint_next_epoch = epoch + 1
                    next_iteration = 0
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
                        next_iteration_in_epoch=next_iteration,
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
