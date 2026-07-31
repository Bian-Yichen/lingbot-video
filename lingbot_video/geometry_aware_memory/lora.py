from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn


DEFAULT_BACKBONE_LORA_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class BackboneLoRAConfig:
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_BACKBONE_LORA_TARGETS

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0,1)")
        if not self.target_modules:
            raise ValueError("LoRA target_modules cannot be empty")
        if any(not name or "." in name for name in self.target_modules):
            raise ValueError(
                "LoRA target_modules must be non-empty leaf module names"
            )


@dataclass(frozen=True)
class BackboneLoRASummary:
    module_names: tuple[str, ...]
    parameter_count: int

    @property
    def module_count(self) -> int:
        return len(self.module_names)


class LoRALinear(nn.Module):
    """A frozen Linear plus a trainable low-rank residual.

    This deliberately keeps the original Linear as ``base_layer`` so loading
    the LingBot checkpoint happens before LoRA injection and no duplicate base
    weights are written to a trainable-components checkpoint.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Linear(
            base_layer.in_features,
            self.rank,
            bias=False,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        self.lora_b = nn.Linear(
            self.rank,
            base_layer.out_features,
            bias=False,
            device=base_layer.weight.device,
            dtype=base_layer.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.base_layer.requires_grad_(False)

    @property
    def weight(self) -> nn.Parameter:
        # LingBot uses ``to_q.weight.dtype`` to select the bulk compute dtype.
        return self.base_layer.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base_layer.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(hidden_states)
        adapter_input = self.dropout(hidden_states).to(self.lora_a.weight.dtype)
        adapter = self.lora_b(self.lora_a(adapter_input))
        return base + adapter.to(base.dtype) * self.scaling


def _normalise_targets(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        targets = tuple(
            token.strip() for token in value.split(",") if token.strip()
        )
    elif isinstance(value, Sequence):
        targets = tuple(str(token).strip() for token in value if str(token).strip())
    else:
        raise TypeError("lora_target_modules must be a list or comma-separated string")
    return targets


def lora_config_from_mapping(config: Mapping[str, Any]) -> BackboneLoRAConfig:
    result = BackboneLoRAConfig(
        rank=int(config.get("lora_rank", 32)),
        alpha=float(config.get("lora_alpha", 32.0)),
        dropout=float(config.get("lora_dropout", 0.0)),
        target_modules=_normalise_targets(
            config.get(
                "lora_target_modules",
                DEFAULT_BACKBONE_LORA_TARGETS,
            )
        ),
    )
    result.validate()
    return result


def inject_backbone_lora(
    backbone: nn.Module,
    config: BackboneLoRAConfig,
) -> BackboneLoRASummary:
    """Freeze ``backbone`` and inject LoRA into selected Linear leaf modules."""

    config.validate()
    backbone.requires_grad_(False)
    replacements: list[tuple[str, nn.Linear]] = []
    targets = set(config.target_modules)
    for module_name, module in backbone.named_modules():
        if not module_name.startswith("blocks."):
            continue
        if isinstance(module, nn.Linear) and module_name.rsplit(".", 1)[-1] in targets:
            replacements.append((module_name, module))

    if not replacements:
        raise ValueError(
            "none of the requested LoRA targets were found in LingBot blocks: "
            f"{sorted(targets)}"
        )

    injected: list[str] = []
    parameter_count = 0
    for module_name, linear in replacements:
        parent_name, child_name = module_name.rsplit(".", 1)
        parent = backbone.get_submodule(parent_name)
        current = getattr(parent, child_name)
        if isinstance(current, LoRALinear):
            raise RuntimeError(f"LoRA is already injected into {module_name}")
        wrapped = LoRALinear(
            linear,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, child_name, wrapped)
        injected.append(module_name)
        parameter_count += sum(
            parameter.numel()
            for parameter in wrapped.parameters()
            if parameter.requires_grad
        )

    return BackboneLoRASummary(
        module_names=tuple(injected),
        parameter_count=parameter_count,
    )


def is_lora_parameter_name(name: str) -> bool:
    return ".lora_a." in name or ".lora_b." in name


def validate_partial_checkpoint_load(
    missing: Sequence[str],
    unexpected: Sequence[str],
    *,
    backbone_train_mode: str,
) -> None:
    """Validate model loading while allowing omitted frozen base weights."""

    if backbone_train_mode == "full":
        allowed_missing: set[str] = set()
    elif backbone_train_mode == "frozen":
        allowed_missing = {
            name for name in missing if name.startswith("backbone.")
        }
    elif backbone_train_mode == "lora":
        allowed_missing = {
            name
            for name in missing
            if name.startswith("backbone.") and not is_lora_parameter_name(name)
        }
    else:
        raise ValueError(
            "backbone_train_mode must be one of full, frozen, or lora"
        )

    invalid_missing = sorted(set(missing) - allowed_missing)
    if invalid_missing or unexpected:
        raise RuntimeError(
            "checkpoint does not match the GIM model: "
            f"missing={invalid_missing[:16]}, "
            f"unexpected={sorted(unexpected)[:16]}"
        )
