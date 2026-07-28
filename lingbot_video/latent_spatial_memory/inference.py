from __future__ import annotations

import contextlib
from typing import Optional

import torch


def _autocast_for(
    device: torch.device,
    dtype: torch.dtype,
):
    if device.type == "cuda" and dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _frame_timesteps(
    timestep: torch.Tensor,
    *,
    batch_size: int,
    frame_count: int,
    device: torch.device,
    transformer_dtype: torch.dtype,
) -> torch.Tensor:
    # Match LingBotVideoPipeline's timestep quantization before the transformer.
    sigma = timestep.float() / 1000.0
    if transformer_dtype in {torch.bfloat16, torch.float16}:
        sigma = sigma.to(transformer_dtype)
    scalar = (sigma * 1000.0).float().reshape(1).to(device)
    output = scalar.expand(batch_size, frame_count).clone()
    # The first target latent is the clean causal overlap from the capture
    # clip, exactly as in latent_memory_training_step.
    output[:, 0] = 0
    return output


@torch.no_grad()
def denoise_latent_memory_chunk(
    model: torch.nn.Module,
    scheduler,
    *,
    clean_prefix: torch.Tensor,
    memory_latents: torch.Tensor,
    memory_visibility: torch.Tensor,
    target_rays: torch.Tensor,
    preceding_latents: torch.Tensor,
    preceding_rays: torch.Tensor,
    reference_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: Optional[torch.Tensor],
    num_inference_steps: int,
    timestep_shift: float,
    generator: torch.Generator,
    show_progress: bool = True,
) -> torch.Tensor:
    """Denoise one training-shaped chunk while keeping its overlap latent clean."""

    if clean_prefix.ndim != 5 or clean_prefix.shape[2] != 1:
        raise ValueError("clean_prefix must be [B,C,1,H,W]")
    if target_rays.ndim != 5:
        raise ValueError("target_rays must be [B,6,T,H,W]")
    batch_size, channels, _, height, width = clean_prefix.shape
    frame_count = target_rays.shape[2]
    if frame_count < 2:
        raise ValueError("inference needs at least two target latent frames")
    if memory_latents.shape != (
        batch_size,
        channels,
        frame_count + preceding_latents.shape[2],
        height,
        width,
    ):
        raise ValueError("memory readout does not align with target and preceding frames")

    device = clean_prefix.device
    transformer_dtype = next(model.parameters()).dtype
    latents = torch.randn(
        batch_size,
        channels,
        frame_count,
        height,
        width,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    prefix = clean_prefix.float()
    latents[:, :, :1] = prefix
    scheduler.set_timesteps(
        num_inference_steps,
        device=device,
        shift=float(timestep_shift),
    )

    prompt_embeds = prompt_embeds.to(device=device, dtype=transformer_dtype)
    if prompt_mask is not None:
        prompt_mask = prompt_mask.to(device=device)
    timesteps = scheduler.timesteps
    if show_progress:
        from tqdm.auto import tqdm

        timesteps = tqdm(timesteps, desc="denoising", leave=False)
    for timestep in timesteps:
        target_timesteps = _frame_timesteps(
            timestep,
            batch_size=batch_size,
            frame_count=frame_count,
            device=device,
            transformer_dtype=transformer_dtype,
        )
        with _autocast_for(device, transformer_dtype):
            velocity = model(
                noisy_latents=latents,
                target_timesteps=target_timesteps,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                memory_latents=memory_latents,
                memory_visibility=memory_visibility,
                target_rays=target_rays,
                preceding_latents=preceding_latents,
                preceding_rays=preceding_rays,
                reference_latents=reference_latents,
            ).velocity.float()
        latents = scheduler.step(
            velocity,
            timestep,
            latents,
            return_dict=False,
            generator=generator,
        )[0]
        # The scheduler must never diffuse or update the causal overlap token.
        latents[:, :, :1] = prefix
    return latents


@torch.no_grad()
def predict_metric_depth(
    model: torch.nn.Module,
    clean_latents: torch.Tensor,
    target_rays: torch.Tensor,
    projected_depth: torch.Tensor,
    memory_visibility: torch.Tensor,
) -> torch.Tensor:
    """Run the training depth head and return metric depth [B,T,H,W]."""

    device = clean_latents.device
    transformer_dtype = next(model.parameters()).dtype
    with _autocast_for(device, transformer_dtype):
        predicted_log_depth = model.predict_log_depth(
            clean_latents,
            target_rays,
            projected_depth,
            memory_visibility,
        )
    return predicted_log_depth.exp().squeeze(1).float()
