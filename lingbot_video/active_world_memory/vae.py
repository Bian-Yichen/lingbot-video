from __future__ import annotations

import contextlib

import torch


def _vae_latent_to_dit(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        vae.config.latents_mean, device=latents.device, dtype=torch.float32
    ).view(1, -1, 1, 1, 1)
    std_inv = (
        1.0
        / torch.tensor(
            vae.config.latents_std, device=latents.device, dtype=torch.float32
        )
    ).view(1, -1, 1, 1, 1)
    return ((latents.float() - mean) * std_inv).to(latents.dtype)


def _dit_latent_to_vae(vae: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(
        vae.config.latents_mean, device=latents.device, dtype=torch.float32
    ).view(1, -1, 1, 1, 1)
    std_inv = (
        1.0
        / torch.tensor(
            vae.config.latents_std, device=latents.device, dtype=torch.float32
        )
    ).view(1, -1, 1, 1, 1)
    return latents.float() / std_inv + mean


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def _distribution_mode(encoded: object) -> torch.Tensor:
    distribution = getattr(encoded, "latent_dist", None)
    if distribution is not None:
        mode = getattr(distribution, "mode", None)
        return mode() if callable(mode) else distribution.mean
    if isinstance(encoded, tuple):
        return encoded[0]
    sample = getattr(encoded, "sample", None)
    return sample if isinstance(sample, torch.Tensor) else encoded


@torch.no_grad()
def encode_target_video(
    vae: torch.nn.Module,
    rgb_uint8: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode a continuous target video with the official causal Wan VAE."""

    video = rgb_uint8.to(device=device, non_blocking=True)
    if video.dtype == torch.uint8:
        video = video.float().div_(127.5).sub_(1.0)
    else:
        video = video.float().mul_(2.0).sub_(1.0)
    with _autocast(device, dtype):
        encoded = vae.encode(video)
        latents = _distribution_mode(encoded)
    return _vae_latent_to_dit(vae, latents).to(dtype)


@torch.no_grad()
def encode_independent_views(
    vae: torch.nn.Module,
    rgb_uint8: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int = 8,
) -> torch.Tensor:
    """Encode each evidence frame as its own one-frame video.

    This deliberately disables temporal compression across unrelated retrieved
    views, so every memory latent retains an exact camera pose.
    """

    if rgb_uint8.ndim != 5:
        raise ValueError("evidence RGB must be (B,N,3,H,W)")
    batch, views = rgb_uint8.shape[:2]
    flat = rgb_uint8.reshape(batch * views, *rgb_uint8.shape[2:])
    outputs: list[torch.Tensor] = []
    for start in range(0, flat.shape[0], chunk_size):
        video = flat[start : start + chunk_size].to(device=device, non_blocking=True)
        if video.dtype == torch.uint8:
            video = video.float().div_(127.5).sub_(1.0)
        else:
            video = video.float().mul_(2.0).sub_(1.0)
        video = video.unsqueeze(2)
        with _autocast(device, dtype):
            encoded = vae.encode(video)
            latents = _distribution_mode(encoded)
        outputs.append(_vae_latent_to_dit(vae, latents).to(dtype)[:, :, 0])
    latent = torch.cat(outputs, dim=0)
    return latent.reshape(batch, views, *latent.shape[1:])


@torch.no_grad()
def decode_video_latents(
    vae: torch.nn.Module,
    latents: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    vae_latents = _dit_latent_to_vae(vae, latents).to(device=device)
    with _autocast(device, dtype):
        decoded = vae.decode(vae_latents)
    frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
    return frames.float().clamp(-1, 1).add(1).div(2)
