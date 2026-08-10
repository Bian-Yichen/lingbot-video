from __future__ import annotations

from dataclasses import dataclass

import torch

from .agent import ActiveMemoryRollout, HierarchicalMemoryIndex
from .model import ActiveWorldMemoryModel


@dataclass(frozen=True)
class ActiveMemoryGeneration:
    latents: torch.Tensor
    rollout: ActiveMemoryRollout
    memory: HierarchicalMemoryIndex


@torch.no_grad()
def denoise_active_chunk(
    model: ActiveWorldMemoryModel,
    scheduler,
    memory: HierarchicalMemoryIndex,
    *,
    target_canvas: torch.Tensor,
    target_camera_tokens: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_prompt_mask: torch.Tensor,
    latent_shape: tuple[int, int, int, int, int],
    fine_view_tokens: torch.Tensor | None,
    fine_patch_tokens: torch.Tensor | None,
    num_inference_steps: int,
    guidance_scale: float,
    shift: float,
    generator: torch.Generator,
    stop_threshold: float,
) -> tuple[torch.Tensor, ActiveMemoryRollout]:
    rollout = model.agent.rollout(
        target_canvas,
        memory,
        deterministic=True,
        stop_threshold=stop_threshold,
    )
    condition = model.make_condition_tokens(
        rollout,
        memory,
        target_camera_tokens,
        fine_view_tokens,
        fine_patch_tokens,
    )
    device = target_canvas.device
    latents = torch.randn(
        latent_shape, device=device, dtype=torch.float32, generator=generator
    )
    scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
    backbone_dtype = next(model.backbone.parameters()).dtype
    for timestep in scheduler.timesteps:
        timestep_batch = timestep.float().reshape(1).to(device)
        with torch.autocast(
            "cuda",
            dtype=backbone_dtype,
            enabled=device.type == "cuda"
            and backbone_dtype in {torch.float16, torch.bfloat16},
        ):
            conditional = model.generator_forward(
                latents,
                timestep_batch,
                prompt_embeds,
                prompt_mask,
                condition,
            ).float()
            if guidance_scale > 1.0:
                unconditional = model.generator_forward(
                    latents,
                    timestep_batch,
                    negative_prompt_embeds,
                    negative_prompt_mask,
                    condition,
                    drop_condition=True,
                ).float()
            else:
                unconditional = None
        if guidance_scale > 1.0:
            assert unconditional is not None
            prediction = unconditional + guidance_scale * (conditional - unconditional)
        else:
            prediction = conditional
        latents = scheduler.step(
            prediction,
            timestep,
            latents,
            generator=generator,
            return_dict=False,
        )[0]
    return latents, rollout
