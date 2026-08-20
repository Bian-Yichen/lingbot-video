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
import torch.nn as nn
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
from lingbot_video.geometry_aware_memory.three_drae import (  # noqa: E402
    ThreeDRAEViewCurriculum,
    WanThreeDRAEModel,
    three_drae_config_from_wan,
)
from lingbot_video.geometry_aware_memory.three_drae_training import (  # noqa: E402,E501
    DINOv2Discriminator,
    ThreeDRAEObjective,
    adaptive_adversarial_weight,
    hinge_discriminator_loss,
    three_drae_training_step,
)
from lingbot_video.geometry_aware_memory.wan_latent_reconstruction import (  # noqa: E402,E501
    WanDecoderBridge,
)
from lingbot_video.geometry_aware_memory.wan_latent_training import (  # noqa: E402
    prepare_wan_latent_batch,
)


logger = logging.getLogger("lingbot_video.train_memory_3drae")


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
            "Train a Wan-latent adaptation of the paper's 3DRAE: frozen Wan "
            "encoder, 12-layer latent fuse neck, 1K scene tokens, 16-layer "
            "ray-query decoder, and frozen/adapted Wan decoder."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--output_dir", default="outputs/memory_3drae")
    parser.add_argument("--item_list", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)

    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--query_blocks", type=int, default=1)
    parser.add_argument("--query_views_start", type=int, default=2)
    parser.add_argument("--query_views_end", type=int, default=8)
    parser.add_argument("--local_window_rgb_frames", type=int, default=81)
    parser.add_argument("--history_views_start_min", type=int, default=2)
    parser.add_argument("--history_views_start_max", type=int, default=4)
    parser.add_argument("--history_views_end_min", type=int, default=12)
    parser.add_argument("--history_views_end_max", type=int, default=32)
    parser.add_argument("--view_curriculum_epochs", type=int, default=8)
    parser.add_argument("--retrieval_rotation_weight", type=float, default=0.25)
    parser.add_argument("--retrieval_temperature", type=float, default=0.25)
    parser.add_argument("--vae_encode_chunk_rgb_frames", type=int, default=16)

    parser.add_argument("--num_memory_tokens", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--encoder_depth", type=int, default=12)
    parser.add_argument("--decoder_depth", type=int, default=16)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument(
        "--latent_batch_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--decoder_noise_tau", type=float, default=0.8)
    parser.add_argument("--view_mask_probability", type=float, default=0.1)
    parser.add_argument("--view_mask_ratio_min", type=float, default=0.6)
    parser.add_argument("--view_mask_ratio_max", type=float, default=0.9)
    parser.add_argument("--decoder_lora_rank", type=int, default=16)
    parser.add_argument("--decoder_lora_alpha", type=float, default=16.0)
    parser.add_argument("--decoder_refiner_hidden_size", type=int, default=64)

    # RGB MSE + LPIPS + delayed GAN is the paper objective.  The optional
    # latent MSE is specific to this Wan-latent adaptation and defaults off.
    parser.add_argument("--latent_loss_weight", type=float, default=0.0)
    parser.add_argument("--rgb_loss_weight", type=float, default=1.0)
    parser.add_argument("--lpips_weight", type=float, default=1.0)
    parser.add_argument("--gan_loss_weight", type=float, default=0.75)
    parser.add_argument(
        "--discriminator_model_name_or_path",
        default="facebook/dinov2-small",
    )
    parser.add_argument("--discriminator_image_size", type=int, default=224)
    parser.add_argument("--discriminator_start_step", type=int, default=50000)
    parser.add_argument("--adversarial_start_step", type=int, default=60000)
    parser.add_argument("--adaptive_gan_weight_max", type=float, default=1.0e4)

    # Preserve the parent branch's two-stage Wan decoder schedule.
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-4)
    parser.add_argument("--discriminator_learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_warmup_steps", type=int, default=8000)
    parser.add_argument("--decoder_warmup_steps", type=int, default=1000)
    parser.add_argument("--discriminator_warmup_steps", type=int, default=8000)
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
        "query_blocks",
        "query_views_start",
        "query_views_end",
        "local_window_rgb_frames",
        "history_views_start_min",
        "history_views_start_max",
        "history_views_end_min",
        "history_views_end_max",
        "view_curriculum_epochs",
        "vae_encode_chunk_rgb_frames",
        "num_memory_tokens",
        "hidden_size",
        "num_heads",
        "encoder_depth",
        "decoder_depth",
        "decoder_lora_rank",
        "decoder_refiner_hidden_size",
        "num_train_epochs",
        "gradient_accumulation_steps",
        "checkpoint_every_iterations",
        "discriminator_image_size",
    )
    for name in positive:
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    for name in (
        "mlp_ratio",
        "decoder_lora_alpha",
        "learning_rate",
        "decoder_learning_rate",
        "discriminator_learning_rate",
        "max_grad_norm",
        "adaptive_gan_weight_max",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    for name in (
        "latent_loss_weight",
        "rgb_loss_weight",
        "lpips_weight",
        "gan_loss_weight",
        "weight_decay",
        "decoder_noise_tau",
        "lr_warmup_steps",
        "decoder_warmup_steps",
        "discriminator_warmup_steps",
        "discriminator_start_step",
        "adversarial_start_step",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} cannot be negative")
    if not 1 <= args.stage1_epochs <= args.num_train_epochs:
        parser.error("stage1_epochs must be in [1,num_train_epochs]")
    if args.hidden_size % args.num_heads:
        parser.error("hidden_size must be divisible by num_heads")
    if args.adversarial_start_step < args.discriminator_start_step:
        parser.error("adversarial_start_step cannot precede discriminator start")
    if args.gan_loss_weight > 0 and not args.discriminator_model_name_or_path:
        parser.error("GAN loss requires discriminator_model_name_or_path")
    curriculum = _view_curriculum(args)
    try:
        curriculum.validate()
        history_min, history_max, query_views = curriculum.values_for_epoch(
            args.view_curriculum_epochs - 1
        )
        GeometryMemorySampleConfig(
            height=args.height,
            width=args.width,
            target_rgb_frames=query_views,
            query_blocks=args.query_blocks,
            local_window_rgb_frames=args.local_window_rgb_frames,
            memory_views_min=history_min,
            memory_views_max=history_max,
            retrieval_rotation_weight=args.retrieval_rotation_weight,
            retrieval_temperature=args.retrieval_temperature,
        ).validate()
    except ValueError as error:
        parser.error(str(error))
    return args


def _view_curriculum(args: argparse.Namespace) -> ThreeDRAEViewCurriculum:
    return ThreeDRAEViewCurriculum(
        history_start_min=args.history_views_start_min,
        history_start_max=args.history_views_start_max,
        history_end_min=args.history_views_end_min,
        history_end_max=args.history_views_end_max,
        query_start=args.query_views_start,
        query_end=args.query_views_end,
        curriculum_epochs=args.view_curriculum_epochs,
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
        path = path / "memory_3drae.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _cosine_multiplier(step: int, start: int, warmup: int, end: int) -> float:
    if step < start:
        return 0.0
    local_step = step - start
    if warmup > 0 and local_step < warmup:
        return float(local_step) / float(max(1, warmup))
    progress = float(local_step - warmup) / float(max(1, end - start - warmup))
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _generator_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    main_warmup_steps: int,
    stage2_start_step: int,
    decoder_warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    schedules = []
    for group in optimizer.param_groups:
        if group.get("group_role") == "wan_decoder_adapter":
            schedules.append(
                lambda step: _cosine_multiplier(
                    step,
                    stage2_start_step,
                    decoder_warmup_steps,
                    total_steps,
                )
            )
        else:
            schedules.append(
                lambda step: _cosine_multiplier(
                    step,
                    0,
                    main_warmup_steps,
                    total_steps,
                )
            )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedules)


def _discriminator_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    start_step: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    active_steps = max(1, total_steps - start_step)
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _cosine_multiplier(
            step,
            0,
            warmup_steps,
            active_steps,
        ),
    )


def _load_wan_components(
    args: argparse.Namespace,
) -> tuple[nn.Module, WanDecoderBridge, tuple[float, ...], tuple[float, ...]]:
    vae = AutoencoderKLWan.from_pretrained(
        args.model_dir,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).eval().requires_grad_(False)
    if vae.config.patch_size is not None:
        raise NotImplementedError("patchified Wan VAE is not supported")
    if int(vae.config.scale_factor_spatial) != 8:
        raise ValueError("3DRAE Wan adapter expects spatial stride 8")
    means = tuple(float(value) for value in vae.config.latents_mean)
    stds = tuple(float(value) for value in vae.config.latents_std)
    bridge = WanDecoderBridge(
        vae.post_quant_conv,
        vae.decoder,
        latent_channels=len(means),
        refiner_hidden_size=args.decoder_refiner_hidden_size,
        lora_rank=args.decoder_lora_rank,
        lora_alpha=args.decoder_lora_alpha,
    )
    vae.post_quant_conv = None
    vae.decoder = None
    return vae, bridge, means, stds


def _set_requires_grad(module: torch.nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    return (
        sum(parameter.numel() for parameter in module.parameters()),
        sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
    )


def _checkpoint_payload(
    model: WanThreeDRAEModel,
    discriminator: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    discriminator_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
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
    if discriminator is not None:
        payload["discriminator"] = {
            name: value.detach().cpu()
            for name, value in discriminator.state_dict().items()
        }
    if discriminator_scheduler is not None:
        payload["discriminator_scheduler"] = discriminator_scheduler.state_dict()
    if epoch_loader_generator_state is not None:
        payload["epoch_loader_generator_state"] = epoch_loader_generator_state.cpu()
    if args.save_optimizer_state:
        payload["optimizer"] = optimizer.state_dict()
        if discriminator_optimizer is not None:
            payload["discriminator_optimizer"] = (
                discriminator_optimizer.state_dict()
            )
    return payload


def _save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    discriminator_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    output_dir: Path,
    **state: Any,
) -> Path:
    checkpoint_dir = output_dir / (
        f"checkpoint-iter-{state['global_iteration']:08d}-"
        f"step-{state['global_step']:08d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_discriminator = (
        accelerator.unwrap_model(discriminator)
        if discriminator is not None
        else None
    )
    accelerator.save(
        _checkpoint_payload(
            accelerator.unwrap_model(model),
            unwrapped_discriminator,
            optimizer,
            discriminator_optimizer,
            scheduler,
            discriminator_scheduler,
            **state,
        ),
        checkpoint_dir / "memory_3drae.pt",
    )
    return checkpoint_dir


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

    curriculum = _view_curriculum(args)
    initial_min, initial_max, initial_queries = curriculum.values_for_epoch(0)
    sample_config = GeometryMemorySampleConfig(
        height=args.height,
        width=args.width,
        target_rgb_frames=initial_queries,
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
    model_config = three_drae_config_from_wan(
        vae_latents_mean=means,
        vae_latents_std=stds,
        image_height=args.height,
        image_width=args.width,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_memory_tokens=args.num_memory_tokens,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        latent_batch_norm=args.latent_batch_norm,
        decoder_noise_tau=args.decoder_noise_tau,
        view_mask_probability=args.view_mask_probability,
        view_mask_ratio_min=args.view_mask_ratio_min,
        view_mask_ratio_max=args.view_mask_ratio_max,
        decoder_lora_rank=args.decoder_lora_rank,
        decoder_lora_alpha=args.decoder_lora_alpha,
        decoder_refiner_hidden_size=args.decoder_refiner_hidden_size,
    )
    model = WanThreeDRAEModel(wan_decoder, model_config)
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    main_parameters = [
        *model.input_projection.parameters(),
        *model.ray_embedding.parameters(),
        *model.memory_encoder.parameters(),
        *model.latent_decoder.parameters(),
    ]
    decoder_parameters = model.wan_decoder.adapter_parameters()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": main_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
                "group_role": "three_drae",
            },
            {
                "params": decoder_parameters,
                "lr": args.decoder_learning_rate,
                "weight_decay": 0.0,
                "group_role": "wan_decoder_adapter",
            },
        ],
        betas=(0.9, 0.95),
    )
    discriminator: DINOv2Discriminator | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    if args.gan_loss_weight > 0:
        discriminator = DINOv2Discriminator(
            args.discriminator_model_name_or_path,
            image_size=args.discriminator_image_size,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        discriminator_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.discriminator_learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )

    if discriminator is None:
        model, optimizer, dataloader = accelerator.prepare(
            model,
            optimizer,
            dataloader,
        )
    else:
        assert discriminator_optimizer is not None
        (
            model,
            discriminator,
            optimizer,
            discriminator_optimizer,
            dataloader,
        ) = accelerator.prepare(
            model,
            discriminator,
            optimizer,
            discriminator_optimizer,
            dataloader,
        )
    vae_encoder.to(accelerator.device)
    steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_optimizer_steps = args.num_train_epochs * steps_per_epoch
    stage2_start_step = args.stage1_epochs * steps_per_epoch
    scheduler_optimizer = getattr(optimizer, "optimizer", optimizer)
    scheduler = _generator_scheduler(
        scheduler_optimizer,
        total_steps=total_optimizer_steps,
        main_warmup_steps=args.lr_warmup_steps,
        stage2_start_step=stage2_start_step,
        decoder_warmup_steps=args.decoder_warmup_steps,
    )
    discriminator_scheduler = None
    if discriminator_optimizer is not None:
        discriminator_scheduler = _discriminator_scheduler(
            getattr(discriminator_optimizer, "optimizer", discriminator_optimizer),
            total_steps=total_optimizer_steps,
            start_step=args.discriminator_start_step,
            warmup_steps=args.discriminator_warmup_steps,
        )
    objective = ThreeDRAEObjective(
        latent_weight=args.latent_loss_weight,
        rgb_weight=args.rgb_loss_weight,
        lpips_weight=args.lpips_weight,
    ).to(accelerator.device)

    unwrapped = accelerator.unwrap_model(model)
    total_parameters, trainable_parameters = _count_parameters(unwrapped)
    patches_per_view = unwrapped.patch_grid[0] * unwrapped.patch_grid[1]
    end_encoder_tokens = (
        args.num_memory_tokens + args.history_views_end_max * patches_per_view
    )
    decoder_tokens = args.num_memory_tokens + patches_per_view
    run_summary = {
        "experiment": "wan_memory_3drae",
        "paper_topology": "12_layer_fuse_1k_scene_tokens_16_layer_ray_query",
        "dataset_items": len(dataset),
        "resolution": [args.height, args.width],
        "history_target_overlap": 0,
        "target_placement": "centered_contiguous_with_history_candidates_on_both_sides",
        "history_curriculum_start": [initial_min, initial_max],
        "history_curriculum_end": [
            args.history_views_end_min,
            args.history_views_end_max,
        ],
        "query_curriculum": [args.query_views_start, args.query_views_end],
        "view_curriculum_epochs": args.view_curriculum_epochs,
        "patches_per_view": patches_per_view,
        "memory_tokens": args.num_memory_tokens,
        "encoder_tokens_at_max_history": end_encoder_tokens,
        "decoder_tokens_per_target_view": decoder_tokens,
        "encoder_attention_warning": "global attention is quadratic in encoder token count",
        "stage_1": "train_3drae_with_frozen_wan_decoder",
        "stage_1_epochs": args.stage1_epochs,
        "stage_2": "continue_3drae_enable_wan_decoder_adapters",
        "loss": "rgb_mse_plus_lpips_plus_delayed_adaptive_gan_optional_latent_mse",
        "discriminator_start_step": args.discriminator_start_step,
        "adversarial_start_step": args.adversarial_start_step,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "total_optimizer_steps": total_optimizer_steps,
    }
    if accelerator.is_main_process:
        logger.info(
            "WAN 3DRAE TRAINING CONFIG\n%s",
            json.dumps(run_summary, indent=2, ensure_ascii=False),
        )
        if end_encoder_tokens > 12000:
            logger.warning(
                "paper-faithful global encoder attention reaches %d tokens "
                "at history_views_max=%d; first validate a smaller history "
                "range because compute grows quadratically",
                end_encoder_tokens,
                args.history_views_end_max,
            )
        (output_dir / "resolved_training_config.json").write_text(
            json.dumps(
                {**vars(args), **run_summary, "model_config": model_config.to_dict()},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    accelerator.init_trackers(
        "wan_memory_3drae",
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
        if checkpoint_payload.get("model_config") != model_config.to_dict():
            raise ValueError(
                "resume checkpoint model_config does not match the current "
                "3DRAE architecture"
            )
        unwrapped.load_experiment_state_dict(checkpoint_payload["model"])
        if discriminator is not None:
            if "discriminator" not in checkpoint_payload:
                raise ValueError(
                    "GAN is enabled but the resume checkpoint has no discriminator"
                )
            accelerator.unwrap_model(discriminator).load_state_dict(
                checkpoint_payload["discriminator"]
            )
        if "optimizer" in checkpoint_payload:
            optimizer.load_state_dict(checkpoint_payload["optimizer"])
        else:
            logger.warning("checkpoint has no generator optimizer state")
        if (
            discriminator_optimizer is not None
            and "discriminator_optimizer" in checkpoint_payload
        ):
            discriminator_optimizer.load_state_dict(
                checkpoint_payload["discriminator_optimizer"]
            )
        scheduler.load_state_dict(checkpoint_payload["scheduler"])
        for parameter_group, learning_rate in zip(
            scheduler_optimizer.param_groups,
            scheduler.get_last_lr(),
            strict=True,
        ):
            parameter_group["lr"] = learning_rate
        if (
            discriminator_scheduler is not None
            and "discriminator_scheduler" in checkpoint_payload
        ):
            discriminator_scheduler.load_state_dict(
                checkpoint_payload["discriminator_scheduler"]
            )
            assert discriminator_optimizer is not None
            for parameter_group, learning_rate in zip(
                discriminator_optimizer.param_groups,
                discriminator_scheduler.get_last_lr(),
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
    if discriminator is not None:
        discriminator.train()
    for epoch in range(start_epoch, args.num_train_epochs):
        stage = 1 if epoch < args.stage1_epochs else 2
        unwrapped.set_training_stage(stage)
        history_min, history_max, query_views = curriculum.values_for_epoch(epoch)
        epoch_sample_config = replace(
            sample_config,
            memory_views_min=history_min,
            memory_views_max=history_max,
            target_rgb_frames=query_views,
        )
        epoch_sample_config.validate()
        dataset.sample_config = epoch_sample_config
        dataset.set_epoch(epoch)
        logger.info(
            "epoch=%d/%d stage=%d scenes=%d history=%d..%d query=%d",
            epoch + 1,
            args.num_train_epochs,
            stage,
            len(dataset),
            history_min,
            history_max,
            epoch_sample_config.query_rgb_frames,
        )
        iteration_offset = resume_iteration_in_epoch if epoch == start_epoch else 0
        if not 0 <= iteration_offset <= len(dataloader):
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
            accumulate_context = (
                accelerator.accumulate(model, discriminator)
                if discriminator is not None
                else accelerator.accumulate(model)
            )
            with accumulate_context:
                step_output = three_drae_training_step(
                    model,
                    prepared,
                    objective=objective,
                    device=accelerator.device,
                    compute_dtype=_dtype(args.mixed_precision),
                )
                metrics = step_output.metrics
                reconstruction_loss = metrics["reconstruction_loss"]
                adversarial_loss = reconstruction_loss.new_zeros(())
                adaptive_weight = reconstruction_loss.new_zeros(())
                generator_loss = reconstruction_loss
                adversarial_active = (
                    discriminator is not None
                    and global_step >= args.adversarial_start_step
                )
                if adversarial_active:
                    assert discriminator is not None
                    _set_requires_grad(discriminator, False)
                    # The discriminator is a differentiable frozen function
                    # for the generator loss.  Bypass its DDP wrapper here:
                    # only gradients to fake RGB are needed, and reducer state
                    # must remain reserved for the later discriminator update.
                    fake_logits = accelerator.unwrap_model(discriminator)(
                        step_output.predicted_rgb
                    )
                    adversarial_loss = -fake_logits.mean()
                    adaptive_weight = adaptive_adversarial_weight(
                        reconstruction_loss,
                        adversarial_loss,
                        unwrapped.last_generator_layer,
                        maximum=args.adaptive_gan_weight_max,
                    )
                    generator_loss = reconstruction_loss + (
                        args.gan_loss_weight
                        * adaptive_weight
                        * adversarial_loss
                    )
                metrics["loss"] = generator_loss
                metrics["generator_adversarial_loss"] = adversarial_loss.detach()
                metrics["adaptive_gan_weight"] = adaptive_weight.detach()
                accelerator.backward(generator_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                discriminator_loss = reconstruction_loss.new_zeros(())
                discriminator_active = (
                    discriminator is not None
                    and global_step >= args.discriminator_start_step
                )
                if discriminator_active:
                    assert discriminator is not None
                    assert discriminator_optimizer is not None
                    _set_requires_grad(discriminator, True)
                    real_logits = discriminator(step_output.target_rgb.detach())
                    fake_logits = discriminator(step_output.predicted_rgb.detach())
                    discriminator_loss = hinge_discriminator_loss(
                        real_logits,
                        fake_logits,
                    )
                    accelerator.backward(discriminator_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            discriminator.parameters(),
                            args.max_grad_norm,
                        )
                    discriminator_optimizer.step()
                    if (
                        accelerator.sync_gradients
                        and discriminator_scheduler is not None
                    ):
                        discriminator_scheduler.step()
                    discriminator_optimizer.zero_grad(set_to_none=True)
                metrics["discriminator_loss"] = discriminator_loss.detach()

            global_iteration += 1
            if accelerator.sync_gradients:
                global_step += 1
            metric_names = (
                "loss",
                "reconstruction_loss",
                "latent_mse",
                "rgb_mse",
                "lpips",
                "psnr",
                "generator_adversarial_loss",
                "adaptive_gan_weight",
                "discriminator_loss",
                "memory_norm",
                "latent_prediction_std",
                "rgb_prediction_std",
                "visible_history_fraction",
            )
            local_metrics = torch.stack(
                [metrics[name].detach().float() for name in metric_names]
            ).unsqueeze(0)
            gathered = accelerator.gather(local_metrics)
            means = gathered.reshape(-1, len(metric_names)).mean(dim=0)
            logged = {
                name: float(value.item())
                for name, value in zip(metric_names, means, strict=True)
            }
            if accelerator.is_main_process:
                logger.info(
                    "scene epoch=%d/%d stage=%d iter=%d/%d item=%s "
                    "step=%d loss=%.6g rgb_mse=%.6g lpips=%.6g psnr=%.3f "
                    "history=%d query=%d visible=%.3f elapsed=%.1fs",
                    epoch + 1,
                    args.num_train_epochs,
                    stage,
                    scene_step + 1,
                    len(dataloader),
                    sample.item_name,
                    global_step,
                    logged["loss"],
                    logged["rgb_mse"],
                    logged["lpips"],
                    logged["psnr"],
                    len(sample.capture_rgb_indices),
                    epoch_sample_config.query_rgb_frames,
                    logged["visible_history_fraction"],
                    time.perf_counter() - started_at,
                )
            if accelerator.sync_gradients:
                tracking = {
                    **{f"train/{key}": value for key, value in logged.items()},
                    "train/stage": stage,
                    "train/history_views_min": history_min,
                    "train/history_views_max": history_max,
                    "train/query_views": query_views,
                    "train/three_drae_lr": optimizer.param_groups[0]["lr"],
                    "train/wan_decoder_lr": optimizer.param_groups[1]["lr"],
                }
                if discriminator_optimizer is not None:
                    tracking["train/discriminator_lr"] = (
                        discriminator_optimizer.param_groups[0]["lr"]
                    )
                accelerator.log(tracking, step=global_step)
            if global_iteration >= next_checkpoint and accelerator.sync_gradients:
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
                        discriminator,
                        optimizer,
                        discriminator_optimizer,
                        scheduler,
                        discriminator_scheduler,
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
                discriminator,
                optimizer,
                discriminator_optimizer,
                scheduler,
                discriminator_scheduler,
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
