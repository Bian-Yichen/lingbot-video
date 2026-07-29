from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import GIMWorldLingBotModel
from .pruning import MIGreedyPruner


@dataclass
class DynamicGIMHistory:
    """Unbounded observation history with bounded GIM recomputation.

    GIM-World is not a recurrent slot updater.  Equation (3) recomputes
    ``m_t=M(H_t)`` after generated observations are appended.  This state
    implements that exact behavior: it keeps history latents/cameras, reruns MI
    pruning, and then reruns the fixed-size memory encoder.
    """

    latents: torch.Tensor  # [1,C,T,H,W], normally kept on CPU
    c2w: torch.Tensor  # [1,T,4,4]
    intrinsics: torch.Tensor  # [1,T,3,3]
    times: torch.Tensor  # [T]

    def __post_init__(self) -> None:
        frames = self.latents.shape[2]
        if self.latents.ndim != 5 or self.latents.shape[0] != 1:
            raise ValueError("latents must be [1,C,T,H,W]")
        if self.c2w.shape != (1, frames, 4, 4):
            raise ValueError("c2w must be [1,T,4,4]")
        if self.intrinsics.shape != (1, frames, 3, 3):
            raise ValueError("intrinsics must be [1,T,3,3]")
        if self.times.shape != (frames,):
            raise ValueError("times must be [T]")

    @property
    def frame_count(self) -> int:
        return int(self.latents.shape[2])

    def append(
        self,
        generated_latents: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        times: torch.Tensor,
    ) -> None:
        if generated_latents.shape[0] != 1:
            raise ValueError("generated_latents must have batch size 1")
        count = generated_latents.shape[2]
        if c2w.shape != (1, count, 4, 4):
            raise ValueError("appended c2w shape mismatch")
        if intrinsics.shape != (1, count, 3, 3):
            raise ValueError("appended intrinsics shape mismatch")
        if times.shape != (count,):
            raise ValueError("appended times shape mismatch")
        target_device = self.latents.device
        self.latents = torch.cat(
            (self.latents, generated_latents.to(target_device)),
            dim=2,
        )
        self.c2w = torch.cat(
            (self.c2w, c2w.to(self.c2w.device)),
            dim=1,
        )
        self.intrinsics = torch.cat(
            (
                self.intrinsics,
                intrinsics.to(self.intrinsics.device),
            ),
            dim=1,
        )
        self.times = torch.cat(
            (self.times, times.to(self.times.device)),
            dim=0,
        )

    @torch.no_grad()
    def build_memory(
        self,
        model: GIMWorldLingBotModel,
        pruner: MIGreedyPruner,
        *,
        budget: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = pruner.select(
            self.c2w[0].cpu(),
            self.times.cpu(),
            budget,
        )
        selected = selected[
            torch.argsort(self.times.cpu()[selected], stable=True)
        ]
        memory = model.build_memory(
            self.latents[:, :, selected].to(device=device, dtype=dtype),
            self.c2w[:, selected].to(device),
            self.intrinsics[:, selected].to(device),
        )
        return memory, selected
