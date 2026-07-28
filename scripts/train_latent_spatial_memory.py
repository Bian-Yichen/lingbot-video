from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.latent_spatial_memory.controlnet import (  # noqa: E402
    LingBotLatentMemoryControlNet,
)
from lingbot_video.latent_spatial_memory.data import (  # noqa: E402
    LongTrajectorySampleConfig,
    RcloneConfig,
    RemoteVipeRoomTourDataset,
)
from lingbot_video.latent_spatial_memory.model import (  # noqa: E402
    LatentMetricDepthHead,
    LingBotVideoLatentMemoryModel,
)
from lingbot_video.latent_spatial_memory.training import (  # noqa: E402
    MemoryTrainingConfig,
    latent_memory_training_step,
)
from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline  # noqa: E402
from lingbot_video.runner import _patch_qwen3vl_from_pretrained  # noqa: E402
from lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModel,
)


logger = logging.getLogger("lingbot_video.train_latent_spatial_memory")


def _config_defaults() -> dict[str, Any]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default=None)
    known, _ = bootstrap.parse_known_args()
    if not known.config:
        return {}
    with Path(known.config).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("training config must be one JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model_dir", required=False)
    parser.add_argument(
        "--dataset_root",
        default="h:bianyichen/AnyReconProDataset_labeled_2/",
    )
    parser.add_argument("--cache_root", default="/tmp/lingbot_latent_memory_cache")
    parser.add_argument("--output_dir", default="outputs/latent_spatial_memory")
    parser.add_argument("--stage", choices=["side_branch", "lora"], default="side_branch")
    parser.add_argument("--init_component_checkpoint", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--prompt", default="An indoor room tour.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--capture_clips", type=int, default=16)
    parser.add_argument("--capture_clip_rgb_frames", type=int, default=9)
    parser.add_argument("--preceding_rgb_frames", type=int, default=8)
    parser.add_argument("--reference_frames", type=int, default=4)
    parser.add_argument("--history_min_frames", type=int, default=256)
    parser.add_argument("--history_max_frames", type=int, default=4096)
    parser.add_argument("--latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--samples_per_item", type=int, default=32)
    parser.add_argument("--min_depth", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--depth_edge_threshold", type=float, default=0.08)
    parser.add_argument("--memory_consistency_threshold", type=float, default=0.15)
    parser.add_argument("--memory_max_points", type=int, default=750_000)
    parser.add_argument("--memory_voxel_size", type=float, default=0.0)
    parser.add_argument("--teacher_memory_probability", type=float, default=0.5)
    parser.add_argument("--memory_update_noise_std", type=float, default=0.05)
    parser.add_argument(
        "--memory_update_dropout_probability",
        type=float,
        default=0.05,
    )
    parser.add_argument("--timestep_shift", type=float, default=5.0)
    parser.add_argument("--depth_loss_weight", type=float, default=0.1)
    parser.add_argument("--text_dropout_probability", type=float, default=0.2)
    parser.add_argument("--control_block_indices", default="0,3,6,9,12,15,18,21")
    parser.add_argument("--capture_encode_batch_size", type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--max_train_steps", type=int, default=100_000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--checkpoint_every", type=int, default=1000)
    parser.add_argument(
        "--save_accelerator_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save the full Accelerator/FSDP state; large but supports exact optimizer resume.",
    )
    parser.add_argument("--item_list", default=None)
    parser.add_argument("--rclone_binary", default="rclone")
    parser.add_argument("--rclone_config", default=None)
    parser.add_argument("--rclone_clear_proxy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rclone_transfers", type=int, default=32)
    parser.add_argument("--rclone_checkers", type=int, default=32)
    parser.set_defaults(**_config_defaults())
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("--model_dir is required (it may be supplied by --config)")
    if args.height % 16 or args.width % 16:
        parser.error("--height and --width must both be multiples of 16")
    if args.latent_frames_per_chunk < 2:
        parser.error("--latent_frames_per_chunk must be at least 2")
    if args.capture_clips < 1:
        parser.error("--capture_clips must be at least 1")
    if not 0 <= args.teacher_memory_probability <= 1:
        parser.error("--teacher_memory_probability must be in [0,1]")
    if not 0 <= args.memory_update_dropout_probability <= 1:
        parser.error("--memory_update_dropout_probability must be in [0,1]")
    if args.train_batch_size < 1:
        parser.error("--train_batch_size must be at least 1")
    if (
        args.stage == "lora"
        and not args.init_component_checkpoint
        and not args.resume_from_checkpoint
    ):
        parser.error(
            "stage=lora requires --init_component_checkpoint from stage 1 "
            "or --resume_from_checkpoint"
        )
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.stage == "side_branch" else 1e-4
    return args


def _dtype(name: str) -> torch.dtype:
    return {
        "no": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def _read_item_list(path: str | None) -> list[str] | None:
    if not path:
        return None
    return [
        line.strip().rstrip("/")
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _load_base_components(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor]:
    compute_dtype = _dtype(args.mixed_precision)
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        args.model_dir,
        subfolder="transformer",
        torch_dtype=compute_dtype,
    )
    dtype_map = {
        "default": compute_dtype,
        "transformer": compute_dtype,
        "text_encoder": compute_dtype,
        "vae": torch.float32,
    }
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
            torch_dtype=dtype_map,
        )
    pipe.text_encoder.to(device)
    with torch.no_grad():
        prompt_embeds, prompt_mask = pipe.encode_prompt(args.prompt, device=device)
    prompt_embeds = prompt_embeds.detach()
    prompt_mask = prompt_mask.detach()
    vae = pipe.vae
    vae.requires_grad_(False)
    vae.eval().to(device)
    transformer = pipe.transformer
    pipe.text_encoder.to("cpu")
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return transformer, vae, prompt_embeds, prompt_mask


def _configure_trainable_model(
    backbone: torch.nn.Module,
    args: argparse.Namespace,
) -> LingBotVideoLatentMemoryModel:
    indices = tuple(
        int(value.strip())
        for value in args.control_block_indices.split(",")
        if value.strip()
    )
    controlnet = LingBotLatentMemoryControlNet.from_backbone(backbone, indices)
    depth_head = LatentMetricDepthHead(int(backbone.config.in_channels))
    backbone.requires_grad_(False)
    if args.stage == "lora":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["to_q", "to_k", "to_v", "to_out"],
            bias="none",
        )
        backbone = get_peft_model(backbone, lora)
    model = LingBotVideoLatentMemoryModel(backbone, controlnet, depth_head)
    if args.init_component_checkpoint:
        payload = torch.load(
            args.init_component_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        state = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            "loaded component checkpoint %s missing=%d unexpected=%d",
            args.init_component_checkpoint,
            len(missing),
            len(unexpected),
        )
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    return model


def _save_trainable_components(
    accelerator,
    model: torch.nn.Module,
    output_dir: Path,
    *,
    step: int,
    args: argparse.Namespace,
) -> None:
    unwrapped = accelerator.unwrap_model(model)
    trainable_names = {
        name
        for name, parameter in unwrapped.named_parameters()
        if parameter.requires_grad
    }
    # Accelerator handles the required all-rank FSDP state-dict collective.
    state = accelerator.get_state_dict(model)
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered = {
        name: value.detach().cpu()
        for name, value in state.items()
        if name in trainable_names
        or name.startswith("controlnet.")
        or name.startswith("depth_head.")
    }
    torch.save(
        {
            "model": filtered,
            "step": int(step),
            "stage": args.stage,
            "control_block_indices": args.control_block_indices,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
        },
        output_dir / "trainable_components.pt",
    )
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resume_step(path: str | None) -> int:
    if not path:
        return 0
    accelerator_state = Path(path)
    candidates = (
        accelerator_state / "trainable_components.pt",
        accelerator_state.parent / "trainable_components.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            return int(payload.get("step", 0))
    checkpoint_name = (
        accelerator_state.parent.name
        if accelerator_state.name == "accelerator_state"
        else accelerator_state.name
    )
    if checkpoint_name.startswith("checkpoint-"):
        try:
            return int(checkpoint_name.removeprefix("checkpoint-"))
        except ValueError:
            pass
    raise ValueError(
        "could not infer the optimizer step for --resume_from_checkpoint; "
        "keep trainable_components.pt next to accelerator_state"
    )


def main() -> None:
    args = parse_args()
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import get_cosine_schedule_with_warmup

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )
    set_seed(args.seed, device_specific=True)
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    backbone, vae, prompt_embeds, prompt_mask = _load_base_components(
        args,
        accelerator.device,
    )
    model = _configure_trainable_model(backbone, args)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("no trainable model parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        betas=(0.0, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    sample_config = LongTrajectorySampleConfig(
        height=args.height,
        width=args.width,
        capture_clips=args.capture_clips,
        capture_clip_rgb_frames=args.capture_clip_rgb_frames,
        preceding_rgb_frames=args.preceding_rgb_frames,
        reference_frames=args.reference_frames,
        history_min_frames=args.history_min_frames,
        history_max_frames=args.history_max_frames,
        latent_frames_per_chunk=args.latent_frames_per_chunk,
        samples_per_item=args.samples_per_item,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    rclone = RcloneConfig(
        binary=args.rclone_binary,
        config_path=args.rclone_config,
        clear_proxy=args.rclone_clear_proxy,
        transfers=args.rclone_transfers,
        checkers=args.rclone_checkers,
    )
    dataset = RemoteVipeRoomTourDataset(
        args.dataset_root,
        args.cache_root,
        sample_config,
        rclone=rclone,
        item_list=_read_item_list(args.item_list),
        seed=args.seed,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    dataloader_options = {
        "batch_size": args.train_batch_size,
        "num_workers": args.dataloader_workers,
        "pin_memory": True,
        "persistent_workers": args.dataloader_workers > 0,
    }
    if args.dataloader_workers > 0:
        # One sample already contains more than one hundred decoded RGB-D
        # frames; avoid the DataLoader default of two prefetched batches/worker.
        dataloader_options["prefetch_factor"] = 1
    dataloader = DataLoader(dataset, **dataloader_options)
    # The iterable dataset already shards items by rank and worker so that
    # processes download different large room-tour items.  Preparing the
    # DataLoader as well would apply a second Accelerate shard and waste data.
    model, optimizer, scheduler = accelerator.prepare(
        model,
        optimizer,
        scheduler,
    )
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)

    training_config = MemoryTrainingConfig(
        latent_frames_per_chunk=args.latent_frames_per_chunk,
        depth_loss_weight=args.depth_loss_weight,
        teacher_memory_probability=args.teacher_memory_probability,
        memory_update_noise_std=args.memory_update_noise_std,
        memory_update_dropout_probability=args.memory_update_dropout_probability,
        timestep_shift=args.timestep_shift,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        depth_edge_threshold=args.depth_edge_threshold,
        memory_consistency_threshold=args.memory_consistency_threshold,
        memory_max_points=args.memory_max_points,
        memory_voxel_size=args.memory_voxel_size,
        capture_encode_batch_size=args.capture_encode_batch_size,
    )
    accelerator.init_trackers(
        "lingbot-latent-spatial-memory",
        config={
            key: value
            for key, value in vars(args).items()
            if isinstance(value, (str, int, float, bool)) or value is None
        },
    )
    model.train()
    step = _resume_step(args.resume_from_checkpoint)
    data_iterator = iter(dataloader)
    while step < args.max_train_steps:
        batch = next(data_iterator)
        batch = {
            key: value.to(accelerator.device, non_blocking=True)
            if torch.is_tensor(value)
            else value
            for key, value in batch.items()
        }
        with accelerator.accumulate(model):
            with accelerator.autocast():
                output = latent_memory_training_step(
                    model,
                    vae,
                    batch,
                    prompt_embeds,
                    prompt_mask,
                    training_config,
                    text_dropout_probability=args.text_dropout_probability,
                )
            accelerator.backward(output.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if not accelerator.sync_gradients:
            continue
        step += 1
        metrics = {
            "train/loss": float(output.loss.detach().item()),
            "train/flow_loss": float(output.flow_loss.item()),
            "train/depth_loss": float(output.depth_loss.item()),
            "train/readout_error": float(output.readout_error.item()),
            "train/memory_points": float(output.memory_points),
            "train/visible_fraction": float(output.visible_fraction),
            "train/learning_rate": float(scheduler.get_last_lr()[0]),
        }
        accelerator.log(metrics, step=step)
        if accelerator.is_main_process and step % args.log_every == 0:
            logger.info("step=%d %s", step, " ".join(f"{k}={v:.6g}" for k, v in metrics.items()))
        if step % args.checkpoint_every == 0:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{step:08d}"
            if args.save_accelerator_state:
                accelerator.save_state(str(checkpoint_dir / "accelerator_state"))
            _save_trainable_components(
                accelerator,
                model,
                checkpoint_dir,
                step=step,
                args=args,
            )
        accelerator.wait_for_everyone()

    final_dir = Path(args.output_dir) / "final"
    if args.save_accelerator_state:
        accelerator.save_state(str(final_dir / "accelerator_state"))
    _save_trainable_components(
        accelerator,
        model,
        final_dir,
        step=step,
        args=args,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
