from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A minimal LoRA wrapper that preserves LingBot's Linear interface."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0,1)")
        self.base = base.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_down = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_up = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))

    @property
    def weight(self) -> nn.Parameter:
        # LingBot queries this dtype to choose its bulk-compute boundary.
        return self.base.weight

    @property
    def bias(self) -> nn.Parameter | None:
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        update = F.linear(
            F.linear(self.dropout(value), self.lora_down),
            self.lora_up,
        )
        return base + update.to(base.dtype) * self.scale


@dataclass(frozen=True)
class LoRAInjectionReport:
    modules: tuple[str, ...]
    parameters: int


def inject_lora(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_suffixes: Iterable[str],
) -> LoRAInjectionReport:
    """Replace selected Linear layers without requiring PEFT at runtime."""

    suffixes = tuple(str(value) for value in target_suffixes)
    if not suffixes:
        raise ValueError("at least one LoRA target suffix is required")
    replacements: list[tuple[str, nn.Linear]] = [
        (name, child)
        for name, child in module.named_modules()
        if isinstance(child, nn.Linear)
        and not isinstance(child, LoRALinear)
        and any(name.endswith(suffix) for suffix in suffixes)
    ]
    if not replacements:
        raise ValueError(f"no Linear modules matched LoRA targets {suffixes}")
    names: list[str] = []
    parameters = 0
    for name, child in replacements:
        parts = name.split(".")
        parent = module
        for part in parts[:-1]:
            parent = getattr(parent, part)
        wrapped = LoRALinear(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        setattr(parent, parts[-1], wrapped)
        names.append(name)
        parameters += wrapped.lora_down.numel() + wrapped.lora_up.numel()
    return LoRAInjectionReport(tuple(names), parameters)


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if ".lora_down" in name or ".lora_up" in name
    }


def load_lora_state_dict(
    module: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    expected = set(lora_state_dict(module))
    received = set(state)
    if expected != received:
        raise RuntimeError(
            "LoRA checkpoint mismatch: "
            f"missing={sorted(expected - received)[:20]}, "
            f"unexpected={sorted(received - expected)[:20]}"
        )
    current = module.state_dict()
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))
