from __future__ import annotations

import argparse
import contextlib
import gc
import json
import logging
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_video.active_world_memory.data import (  # noqa: E402
    ActiveMemorySample,
    ActiveMemorySampleConfig,
    LocalRoomTourDataset,
)
from lingbot_video.active_world_memory.model import (  # noqa: E402
    ActiveWorldMemoryConfig,
    ActiveWorldMemoryModel,
)
from lingbot_video.active_world_memory.lora import (  # noqa: E402
    LoRAInjectionReport,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
)
from lingbot_video.active_world_memory.training import (  # noqa: E402
    ActiveWorldTrainingConfig,
    active_world_training_step,
)
from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline  # noqa: E402
from lingbot_video.runner import _patch_qwen3vl_from_pretrained  # noqa: E402
from lingbot_video.transformer_lingbot_video import (  # noqa: E402
    LingBotVideoTransformer3DModel,
)


logger = logging.getLogger("lingbot_video.train_active_world_memory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Active World Memory around LingBot-Video."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stage",
        choices=["retriever", "imitation", "policy", "joint"],
        default=None,
    )
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _dataclass_from_section(cls, section: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {sorted(unknown)}")
    return cls(**section)


def _read_config(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be one JSON object")
    for name in ("model_dir", "dataset_root", "output_dir"):
        if not payload.get(name):
            raise ValueError(f"config is missing {name!r}")
    return payload


def _read_item_list(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [item.strip().rstrip("/") for item in value if item.strip()]
    return [
        line.strip().rstrip("/")
        for line in Path(value).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _first_sample(batch: list[ActiveMemorySample]) -> ActiveMemorySample:
    if len(batch) != 1:
        raise ValueError("Active World Memory currently uses one scene per rank")
    return batch[0]


def _dtype(name: str) -> torch.dtype:
    try:
        return {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]
    except KeyError as exc:
        raise ValueError(f"unsupported mixed precision {name!r}") from exc


def _load_base(
    config: dict[str, Any],
    device: torch.device,
    compute_dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.Tensor, torch.Tensor]:
    transformer = LingBotVideoTransformer3DModel.from_pretrained(
        config["model_dir"], subfolder="transformer", torch_dtype=compute_dtype
    )
    attention_environment = "LINGBOT_QWEN_ATTN_IMPLEMENTATION"
    previous_attention = os.environ.get(attention_environment)
    os.environ[attention_environment] = str(
        config.get("qwen_attn_implementation", "sdpa")
    )
    try:
        patch_context = (
            _patch_qwen3vl_from_pretrained()
            if _patch_qwen3vl_from_pretrained is not None
            else contextlib.nullcontext()
        )
        with patch_context:
            pipe = LingBotVideoPipeline.from_pretrained(
                config["model_dir"],
                transformer=transformer,
                trust_remote_code=True,
                local_files_only=bool(config.get("local_files_only", True)),
                torch_dtype={
                    "default": compute_dtype,
                    "transformer": compute_dtype,
                    "text_encoder": compute_dtype,
                    "vae": torch.float32,
                },
            )
    finally:
        if previous_attention is None:
            os.environ.pop(attention_environment, None)
        else:
            os.environ[attention_environment] = previous_attention
    pipe.text_encoder.to(device)
    with torch.no_grad():
        prompt_embeds, prompt_mask = pipe.encode_prompt(
            config.get("prompt", "An indoor room tour."), device=device
        )
    vae = pipe.vae.requires_grad_(False).eval().to(device)
    backbone = pipe.transformer
    pipe.text_encoder.to("cpu")
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return backbone, vae, prompt_embeds.detach(), prompt_mask.detach()


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def _new_component_state(model: ActiveWorldMemoryModel) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("backbone.")
    }


def _configure_backbone(
    model: ActiveWorldMemoryModel,
    optimization: dict[str, Any],
) -> tuple[str, LoRAInjectionReport | None]:
    mode = str(optimization.get("backbone_train_mode", "lora"))
    if mode not in {"frozen", "lora", "full"}:
        raise ValueError("backbone_train_mode must be 'frozen', 'lora', or 'full'")
    if mode == "full":
        model.backbone.requires_grad_(True)
        return mode, None
    model.backbone.requires_grad_(False)
    if mode == "frozen":
        return mode, None
    if os.environ.get("LINGBOT_FUSED_QKV_LINEAR") == "1":
        raise ValueError(
            "native LoRA requires LINGBOT_FUSED_QKV_LINEAR=0 because the fused "
            "path bypasses the wrapped Q/K/V Linear forwards"
        )
    report = inject_lora(
        model.backbone,
        rank=int(optimization.get("lora_rank", 16)),
        alpha=float(optimization.get("lora_alpha", 16.0)),
        dropout=float(optimization.get("lora_dropout", 0.0)),
        target_suffixes=optimization.get(
            "lora_targets",
            ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out"],
        ),
    )
    return mode, report


def _checkpoint_file(value: str | Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "trainable_components.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _save_checkpoint(
    accelerator: Accelerator,
    model: ActiveWorldMemoryModel,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> Path | None:
    if not accelerator.is_main_process:
        return None
    unwrapped = accelerator.unwrap_model(model)
    checkpoint_dir = output_dir / f"checkpoint-iter-{global_step:08d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "active_world_memory": _new_component_state(unwrapped),
        "active_world_memory_config": unwrapped.active_config.to_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "stage": config["training"]["stage"],
        "config": config,
    }
    optimization = config.get("optimization", {})
    if optimization.get("save_optimizer_state", False):
        payload["optimizer"] = optimizer.state_dict()
    lora = lora_state_dict(unwrapped.backbone)
    if lora:
        payload["backbone_lora"] = lora
    if optimization.get("save_backbone", False):
        payload["backbone"] = {
            name: value.detach().cpu()
            for name, value in unwrapped.backbone.state_dict().items()
            if value.is_floating_point()
        }
    torch.save(payload, checkpoint_dir / "trainable_components.pt")
    (checkpoint_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("saved %s", checkpoint_dir)
    return checkpoint_dir


def _load_checkpoint(
    model: ActiveWorldMemoryModel,
    optimizer: torch.optim.Optimizer | None,
    value: str | Path,
    *,
    load_training_state: bool,
) -> tuple[int, int]:
    payload = torch.load(_checkpoint_file(value), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(
        payload["active_world_memory"], strict=False
    )
    active_missing = [name for name in missing if not name.startswith("backbone.")]
    if active_missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch: missing={active_missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
    if "backbone" in payload:
        model.backbone.load_state_dict(payload["backbone"], strict=False)
    if "backbone_lora" in payload:
        load_lora_state_dict(model.backbone, payload["backbone_lora"])
    if load_training_state and optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if load_training_state:
        return int(payload.get("epoch", 0)), int(payload.get("global_step", 0))
    return 0, 0


def main() -> None:
    cli = parse_args()
    config = _read_config(cli.config)
    config.setdefault("data", {})
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("optimization", {})
    if cli.stage is not None:
        config["training"]["stage"] = cli.stage
    if cli.resume is not None:
        config["resume_from_checkpoint"] = cli.resume
    os.environ["LINGBOT_FUSED_QKV_LINEAR"] = (
        "1" if bool(config.get("fused_qkv_linear", False)) else "0"
    )

    data_config = _dataclass_from_section(ActiveMemorySampleConfig, config["data"])
    model_config = _dataclass_from_section(ActiveWorldMemoryConfig, config["model"])
    training_config = _dataclass_from_section(
        ActiveWorldTrainingConfig, config["training"]
    )
    data_config.validate()
    model_config.validate()
    training_config.validate()

    optimization = config["optimization"]
    mixed_precision = optimization.get("mixed_precision", "bf16")
    gradient_accumulation = int(optimization.get("gradient_accumulation_steps", 1))
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=None if mixed_precision == "no" else mixed_precision,
        gradient_accumulation_steps=gradient_accumulation,
        log_with="tensorboard" if optimization.get("tensorboard", True) else None,
        project_dir=str(Path(config["output_dir"]) / "tensorboard"),
        kwargs_handlers=[ddp],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    set_seed(int(optimization.get("seed", 42)), device_specific=True)
    compute_dtype = _dtype(mixed_precision)

    item_names = _read_item_list(config.get("item_list"))
    dataset = LocalRoomTourDataset(
        config["dataset_root"],
        data_config,
        item_names=item_names,
        seed=int(optimization.get("seed", 42)),
    )
    workers = int(optimization.get("dataloader_workers", 2))
    loader_kwargs: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": workers,
        "collate_fn": _first_sample,
        "pin_memory": False,
        "persistent_workers": False,
    }
    if workers:
        loader_kwargs["prefetch_factor"] = int(
            optimization.get("dataloader_prefetch_factor", 1)
        )
    dataloader = DataLoader(dataset, **loader_kwargs)

    backbone, vae, prompt_embeds, prompt_mask = _load_base(
        config, accelerator.device, compute_dtype
    )
    model = ActiveWorldMemoryModel(backbone, model_config)
    backbone_mode, lora_report = _configure_backbone(model, optimization)
    model.to(accelerator.device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optimization.get("learning_rate", 1e-4)),
        weight_decay=float(optimization.get("weight_decay", 0.0)),
    )
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    start_epoch = 0
    global_step = 0
    resume = config.get("resume_from_checkpoint")
    if resume:
        start_epoch, global_step = _load_checkpoint(
            accelerator.unwrap_model(model),
            optimizer,
            resume,
            load_training_state=bool(
                optimization.get("resume_training_state", False)
            ),
        )

    output_dir = Path(config["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        total, trainable_count = _count_parameters(accelerator.unwrap_model(model))
        logger.info("=== ACTIVE WORLD MEMORY TRAINING ===")
        logger.info("stage=%s base=%s", training_config.stage, config["model_dir"])
        logger.info(
            "dataset=%s scenes=%d/%d skipped_short=%d",
            config["dataset_root"],
            len(dataset),
            dataset.total_items,
            dataset.skipped_short_items,
        )
        logger.info(
            "target=%d blocks=%d capture_span=%d..%d candidates=%d..%d",
            data_config.target_rgb_frames,
            data_config.query_blocks,
            data_config.capture_span_min_frames,
            data_config.capture_span_max_frames,
            data_config.candidate_views_min,
            data_config.candidate_views_max,
        )
        logger.info(
            "parameters total=%s trainable=%s backbone=%s world_size=%d",
            f"{total:,}",
            f"{trainable_count:,}",
            backbone_mode,
            accelerator.num_processes,
        )
        if lora_report is not None:
            logger.info(
                "LoRA modules=%d parameters=%s rank=%d alpha=%g targets=%s",
                len(lora_report.modules),
                f"{lora_report.parameters:,}",
                int(optimization.get("lora_rank", 16)),
                float(optimization.get("lora_alpha", 16.0)),
                optimization.get("lora_targets"),
            )
        logger.info(
            "workers=%d prefetch=%s precision=%s lr=%g",
            workers,
            loader_kwargs.get("prefetch_factor", "disabled"),
            mixed_precision,
            optimizer.param_groups[0]["lr"],
        )
        accelerator.init_trackers(
            "active_world_memory",
            config={
                "stage": training_config.stage,
                "scenes": len(dataset),
                "world_size": accelerator.num_processes,
                "trainable_parameters": trainable_count,
            },
        )

    epochs = int(optimization.get("num_train_epochs", 20))
    log_every = int(optimization.get("log_every", 1))
    checkpoint_every = int(optimization.get("checkpoint_every_iterations", 200))
    max_grad_norm = float(optimization.get("max_grad_norm", 1.0))
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        model.train()
        if backbone_mode == "frozen":
            accelerator.unwrap_model(model).backbone.eval()
        vae.eval()
        for iteration, sample in enumerate(dataloader, start=1):
            started = time.perf_counter()
            with accelerator.accumulate(model):
                output = active_world_training_step(
                    model,
                    vae,
                    sample,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    config=training_config,
                    device=accelerator.device,
                    compute_dtype=compute_dtype,
                )
                accelerator.backward(output["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1

            local_metrics = {
                f"train/{name}": float(value.detach().float().item())
                for name, value in output.items()
            }
            local_metrics["train/learning_rate"] = float(
                optimizer.param_groups[0]["lr"]
            )
            local_metrics["train/seconds"] = time.perf_counter() - started
            metric_names = list(output)
            metric_values = torch.stack(
                [output[name].detach().float().reshape(()) for name in metric_names]
            )
            reduced_values = accelerator.reduce(metric_values, reduction="mean")
            reduced_metrics = {
                f"train/{name}": float(value.item())
                for name, value in zip(
                    metric_names, reduced_values, strict=True
                )
            }
            reduced_metrics["train/learning_rate"] = local_metrics[
                "train/learning_rate"
            ]
            reduced_metrics["train/seconds"] = float(
                accelerator.reduce(
                    torch.tensor(
                        local_metrics["train/seconds"], device=accelerator.device
                    ),
                    reduction="max",
                ).item()
            )
            should_log = iteration % log_every == 0
            if should_log and optimization.get("log_all_ranks", True):
                logger.info(
                    "scene rank=%d epoch=%d/%d iter=%d/%d step=%d item=%s "
                    "loss=%.6f flow=%.6f retrieval=%.6f critic=%.6f "
                    "imitation=%.6f policy=%.6f selected=%.2f reward=%.5f time=%.1fs",
                    accelerator.process_index,
                    epoch + 1,
                    epochs,
                    iteration,
                    len(dataloader),
                    global_step,
                    sample.item_name,
                    local_metrics["train/loss"],
                    local_metrics["train/flow_loss"],
                    local_metrics["train/retrieval_loss"],
                    local_metrics["train/critic_loss"],
                    local_metrics["train/imitation_loss"],
                    local_metrics["train/policy_loss"],
                    local_metrics["train/selected_views"],
                    local_metrics["train/information_gain_reward"],
                    local_metrics["train/seconds"],
                )
            if accelerator.is_main_process and should_log:
                logger.info(
                    "global epoch=%d/%d iter=%d/%d step=%d "
                    "loss=%.6f flow=%.6f retrieval=%.6f critic=%.6f "
                    "imitation=%.6f policy=%.6f selected=%.2f reward=%.5f time=%.1fs",
                    epoch + 1,
                    epochs,
                    iteration,
                    len(dataloader),
                    global_step,
                    reduced_metrics["train/loss"],
                    reduced_metrics["train/flow_loss"],
                    reduced_metrics["train/retrieval_loss"],
                    reduced_metrics["train/critic_loss"],
                    reduced_metrics["train/imitation_loss"],
                    reduced_metrics["train/policy_loss"],
                    reduced_metrics["train/selected_views"],
                    reduced_metrics["train/information_gain_reward"],
                    reduced_metrics["train/seconds"],
                )
                accelerator.log(reduced_metrics, step=global_step)
            if (
                accelerator.sync_gradients
                and checkpoint_every > 0
                and global_step % checkpoint_every == 0
            ):
                accelerator.wait_for_everyone()
                _save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    output_dir,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                )
                accelerator.wait_for_everyone()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info("completed epoch=%d/%d", epoch + 1, epochs)

    _save_checkpoint(
        accelerator,
        model,
        optimizer,
        output_dir,
        epoch=epochs,
        global_step=global_step,
        config=config,
    )
    accelerator.end_training()


if __name__ == "__main__":
    main()
