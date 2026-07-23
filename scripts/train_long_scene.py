#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader

from lingbot_video.long_scene.config import LongSceneConfig
from lingbot_video.long_scene.data import (
    SyntheticLongSceneDataset,
    load_dataset_factory,
    validate_scene_batch,
)
from lingbot_video.long_scene.model import LongSceneWorldModel
from lingbot_video.long_scene.training import compute_long_scene_training_loss
from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LingBot-Video with dynamic recurrent long-scene memory."
        )
    )
    parser.add_argument("--model_dir")
    parser.add_argument("--transformer_subfolder", default="transformer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_factory")
    parser.add_argument("--dataset_config")
    parser.add_argument("--synthetic_smoke_data", action="store_true")
    parser.add_argument("--tiny_smoke_model", action="store_true")

    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--backbone_learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument(
        "--init_from",
        help="Load trainable weights only, for switching between training stages.",
    )
    parser.add_argument("--resume")

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=("to_q", "to_k", "to_v", "to_out"),
    )
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    parser.add_argument("--smoke_length", type=int, default=8)
    parser.add_argument("--smoke_latent_frames", type=int, default=3)
    parser.add_argument("--smoke_latent_height", type=int, default=8)
    parser.add_argument("--smoke_latent_width", type=int, default=12)
    return parser.parse_args()


def tiny_backbone() -> LingBotVideoTransformer3DModel:
    return LingBotVideoTransformer3DModel(
        patch_size=(1, 2, 2),
        in_channels=4,
        out_channels=4,
        hidden_size=64,
        num_attention_heads=4,
        depth=2,
        intermediate_size=128,
        text_dim=32,
        freq_dim=32,
        axes_dims=(4, 6, 6),
        axes_lens=(512, 128, 128),
        num_experts=0,
    )


def load_backbone(args: argparse.Namespace) -> LingBotVideoTransformer3DModel:
    if args.tiny_smoke_model:
        return tiny_backbone()
    if not args.model_dir:
        raise ValueError("--model_dir is required unless --tiny_smoke_model is used")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }[args.mixed_precision]
    return LingBotVideoTransformer3DModel.from_pretrained(
        args.model_dir,
        subfolder=args.transformer_subfolder,
        torch_dtype=dtype,
    )


def configure_backbone(
    backbone: LingBotVideoTransformer3DModel,
    args: argparse.Namespace,
) -> torch.nn.Module:
    if not args.full_finetune:
        backbone.requires_grad_(False)
    if args.gradient_checkpointing:
        backbone.enable_gradient_checkpointing()
    if args.lora_rank > 0:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(args.lora_target_modules),
            bias="none",
        )
        backbone = get_peft_model(backbone, lora_config)
    return backbone


def build_dataset(
    args: argparse.Namespace,
    config: LongSceneConfig,
    model: LongSceneWorldModel,
):
    if args.synthetic_smoke_data:
        return SyntheticLongSceneDataset(
            length=args.smoke_length,
            capture_frames=max(
                config.capture_chunk_size + 1,
                2 * config.capture_chunk_size,
            ),
            latent_channels=model.latent_channels,
            latent_frames=args.smoke_latent_frames,
            latent_height=args.smoke_latent_height,
            latent_width=args.smoke_latent_width,
            text_dim=model.text_dim,
        )
    if not args.dataset_factory:
        raise ValueError(
            "Provide --dataset_factory module:function or use --synthetic_smoke_data"
        )
    return load_dataset_factory(args.dataset_factory, args.dataset_config)


def optimizer_groups(
    model: LongSceneWorldModel,
    args: argparse.Namespace,
) -> list[dict]:
    memory_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_parameters.append(parameter)
        else:
            memory_parameters.append(parameter)
    groups = []
    if memory_parameters:
        groups.append(
            {
                "params": memory_parameters,
                "lr": args.learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    if backbone_parameters:
        groups.append(
            {
                "params": backbone_parameters,
                "lr": args.backbone_learning_rate,
                "weight_decay": args.weight_decay,
            }
        )
    if not groups:
        raise ValueError("No trainable parameters were selected")
    return groups


def cosine_schedule(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if name in trainable
        or any(name.startswith(parameter_name + ".") for parameter_name in trainable)
    }


def save_checkpoint(
    accelerator: Accelerator,
    model: LongSceneWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    output_dir: Path,
    step: int,
    config: LongSceneConfig,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    checkpoint_dir = output_dir / f"checkpoint-{step:07d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    torch.save(
        {
            "step": step,
            "trainable_model": trainable_state_dict(unwrapped),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        checkpoint_dir / "training_state.pt",
    )
    config.save_json(checkpoint_dir / "long_scene_config.json")
    backbone = unwrapped.backbone
    if hasattr(backbone, "save_pretrained") and hasattr(backbone, "peft_config"):
        backbone.save_pretrained(checkpoint_dir / "backbone_lora")


def load_checkpoint(
    accelerator: Accelerator,
    model: LongSceneWorldModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: str,
) -> int:
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = accelerator.unwrap_model(model).load_state_dict(
        state["trainable_model"], strict=False
    )
    unexpected = [
        name for name in unexpected if not name.endswith("rotary_emb.inv_freq")
    ]
    if unexpected:
        raise ValueError(f"Unexpected checkpoint parameters: {unexpected}")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if accelerator.is_main_process:
        print(
            f"Resumed step {state['step']} with {len(missing)} frozen/base "
            "parameters intentionally absent from the trainable checkpoint."
        )
    return int(state["step"])


def load_trainable_weights(model: LongSceneWorldModel, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)
    trainable_model = state.get("trainable_model", state)
    _, unexpected = model.load_state_dict(trainable_model, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected initialization parameters: {unexpected}")


def main() -> None:
    args = parse_args()
    if args.init_from and args.resume:
        raise ValueError("--init_from and --resume are mutually exclusive")
    config = LongSceneConfig.from_json(args.config)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed, device_specific=True)

    backbone = configure_backbone(load_backbone(args), args)
    model = LongSceneWorldModel(backbone, config)
    if args.init_from:
        load_trainable_weights(model, args.init_from)
        if accelerator.is_main_process:
            print(f"Initialized trainable weights from {args.init_from}")
    dataset = build_dataset(args, config, model)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    optimizer = torch.optim.AdamW(
        optimizer_groups(model, args),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(step, args.warmup_steps, args.max_steps),
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            accelerator, model, optimizer, scheduler, args.resume
        )
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "training_args.json").open(
            "w", encoding="utf-8"
        ) as stream:
            json.dump(vars(args), stream, indent=2, sort_keys=True)
            stream.write("\n")
        config.save_json(output_dir / "long_scene_config.json")

    model.train()
    step = start_step
    while step < args.max_steps:
        for batch in dataloader:
            validate_scene_batch(batch)
            with accelerator.accumulate(model):
                output = compute_long_scene_training_loss(model, batch, config)
                accelerator.backward(output.loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                step += 1
                if accelerator.is_main_process and step % args.log_every == 0:
                    values = {
                        name: float(value.detach())
                        for name, value in output.losses.items()
                    }
                    values["lr"] = scheduler.get_last_lr()[0]
                    print(f"step={step} {json.dumps(values, sort_keys=True)}")
                if args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(
                        accelerator,
                        model,
                        optimizer,
                        scheduler,
                        output_dir,
                        step,
                        config,
                    )
                if step >= args.max_steps:
                    break

    save_checkpoint(
        accelerator,
        model,
        optimizer,
        scheduler,
        output_dir,
        step,
        config,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
