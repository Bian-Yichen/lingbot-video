from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import RoomTourSample, VipeRoomTourItem
from .reconstruction_training import prepare_reconstruction_batch
from .training import encode_wan_frames_independently
from .wan_latent_reconstruction import GIMWanLatentReconstructionModel


@dataclass
class PreparedWanLatentBatch:
    history_latents: torch.Tensor
    history_c2w: torch.Tensor
    history_intrinsics: torch.Tensor
    target_latents: torch.Tensor
    target_rgb: torch.Tensor
    target_c2w: torch.Tensor
    target_intrinsics: torch.Tensor


@torch.no_grad()
def prepare_wan_latent_batch(
    sample: RoomTourSample,
    *,
    vae: torch.nn.Module,
    vae_encode_chunk_rgb_frames: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> PreparedWanLatentBatch:
    prepared = prepare_reconstruction_batch(
        sample,
        vae=vae,
        vae_encode_chunk_rgb_frames=vae_encode_chunk_rgb_frames,
        device=device,
        compute_dtype=compute_dtype,
    )
    target_indices = tuple(
        index for block in sample.query_rgb_blocks for index in block
    )
    preloaded = sample.preloaded_inputs
    item = (
        None
        if preloaded is not None
        else VipeRoomTourItem(Path(sample.local_root))
    )
    target_video = (
        torch.cat(preloaded.query_rgb_uint8_blocks, dim=1)
        if preloaded is not None
        else None
    )
    target_latents = encode_wan_frames_independently(
        vae,
        item,
        target_indices,
        sample.image_hw,
        read_chunk_rgb_frames=vae_encode_chunk_rgb_frames,
        device=device,
        dtype=compute_dtype,
        preloaded_video=target_video,
    )
    return PreparedWanLatentBatch(
        history_latents=prepared.history_latents,
        history_c2w=prepared.history_c2w,
        history_intrinsics=prepared.history_intrinsics,
        target_latents=target_latents,
        target_rgb=prepared.target_rgb,
        target_c2w=prepared.target_c2w,
        target_intrinsics=prepared.target_intrinsics,
    )


class WanLatentReconstructionObjective(nn.Module):
    def __init__(
        self,
        *,
        latent_weight: float,
        rgb_weight: float,
        lpips_weight: float,
    ) -> None:
        super().__init__()
        self.latent_weight = float(latent_weight)
        self.rgb_weight = float(rgb_weight)
        self.lpips_weight = float(lpips_weight)
        if min(self.latent_weight, self.rgb_weight, self.lpips_weight) < 0:
            raise ValueError("reconstruction loss weights cannot be negative")
        if self.latent_weight == self.rgb_weight == self.lpips_weight == 0:
            raise ValueError("at least one reconstruction loss must be enabled")
        self.perceptual: nn.Module | None = None
        if self.lpips_weight > 0:
            try:
                import lpips
            except ImportError as error:
                raise ImportError(
                    "LPIPS is enabled. Install with "
                    "`pip install -e '.[memory-reconstruction]'`."
                ) from error
            self.perceptual = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(
        self,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor,
        predicted_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if predicted_latents.shape != target_latents.shape:
            raise ValueError(
                "predicted/target latent shape mismatch: "
                f"{predicted_latents.shape} vs {target_latents.shape}"
            )
        if predicted_rgb.shape != target_rgb.shape:
            raise ValueError(
                f"predicted/target RGB mismatch: {predicted_rgb.shape} "
                f"vs {target_rgb.shape}"
            )
        latent_mse = F.mse_loss(
            predicted_latents.float(),
            target_latents.float(),
        )
        rgb_mse = F.mse_loss(predicted_rgb.float(), target_rgb.float())
        if self.perceptual is None:
            perceptual = rgb_mse.new_zeros(())
        else:
            batch, views, channels, height, width = predicted_rgb.shape
            prediction_2d = predicted_rgb.float().reshape(
                batch * views,
                channels,
                height,
                width,
            )
            target_2d = target_rgb.float().reshape_as(prediction_2d)
            perceptual = self.perceptual(
                prediction_2d.mul(2.0).sub(1.0),
                target_2d.mul(2.0).sub(1.0),
            ).mean()
        loss = (
            self.latent_weight * latent_mse
            + self.rgb_weight * rgb_mse
            + self.lpips_weight * perceptual
        )
        psnr = -10.0 * torch.log10(rgb_mse.detach().clamp_min(1e-12))
        return {
            "loss": loss,
            "latent_mse": latent_mse.detach(),
            "rgb_mse": rgb_mse.detach(),
            "lpips": perceptual.detach(),
            "psnr": psnr,
        }


def wan_latent_training_step(
    model: GIMWanLatentReconstructionModel,
    batch: PreparedWanLatentBatch,
    *,
    objective: WanLatentReconstructionObjective,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    target_latents = batch.target_latents.to(
        device=device,
        dtype=compute_dtype,
    ).permute(0, 2, 1, 3, 4)
    target_rgb = batch.target_rgb.to(device=device, dtype=compute_dtype)
    predicted_latents, predicted_rgb, memory = model(
        batch.history_latents.to(device=device, dtype=compute_dtype),
        batch.history_c2w.to(device),
        batch.history_intrinsics.to(device),
        batch.target_c2w.to(device),
        batch.target_intrinsics.to(device),
    )
    output = objective(
        predicted_latents,
        target_latents,
        predicted_rgb,
        target_rgb,
    )
    output["memory_norm"] = memory.detach().float().norm(dim=-1).mean()
    output["latent_prediction_std"] = predicted_latents.detach().float().std()
    output["rgb_prediction_std"] = predicted_rgb.detach().float().std()
    return output
