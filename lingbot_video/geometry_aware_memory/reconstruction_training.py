from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import RoomTourSample, VipeRoomTourItem
from .memory_reconstruction import GIMMemoryReconstructionModel
from .training import encode_wan_frames_independently


@dataclass
class PreparedReconstructionBatch:
    history_latents: torch.Tensor
    history_c2w: torch.Tensor
    history_intrinsics: torch.Tensor
    target_rgb: torch.Tensor
    target_c2w: torch.Tensor
    target_intrinsics: torch.Tensor


@torch.no_grad()
def prepare_reconstruction_batch(
    sample: RoomTourSample,
    *,
    vae: torch.nn.Module,
    vae_encode_chunk_rgb_frames: int,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> PreparedReconstructionBatch:
    """Encode history only; held-out RGB targets never pass through the VAE."""

    preloaded = sample.preloaded_inputs
    item = (
        None
        if preloaded is not None
        else VipeRoomTourItem(Path(sample.local_root))
    )
    history_latents = encode_wan_frames_independently(
        vae,
        item,
        sample.capture_rgb_indices,
        sample.image_hw,
        read_chunk_rgb_frames=vae_encode_chunk_rgb_frames,
        device=device,
        dtype=compute_dtype,
        preloaded_video=(
            preloaded.capture_rgb_uint8 if preloaded is not None else None
        ),
    )
    if preloaded is None:
        assert item is not None
        history_c2w, history_intrinsics = item.cameras(
            sample.capture_rgb_indices,
            sample.image_hw,
            origin_index=sample.capture_start,
        )
        target_videos = [
            item.read_video_uint8(indices, sample.image_hw)
            for indices in sample.query_rgb_blocks
        ]
        target_cameras = [
            item.cameras(
                indices,
                sample.image_hw,
                origin_index=sample.capture_start,
            )
            for indices in sample.query_rgb_blocks
        ]
        target_c2w = torch.cat([value[0] for value in target_cameras], dim=0)
        target_intrinsics = torch.cat(
            [value[1] for value in target_cameras],
            dim=0,
        )
    else:
        history_c2w = preloaded.capture_c2w
        history_intrinsics = preloaded.capture_intrinsics
        target_videos = list(preloaded.query_rgb_uint8_blocks)
        target_c2w = torch.cat(preloaded.query_c2w_blocks, dim=0)
        target_intrinsics = torch.cat(
            preloaded.query_intrinsics_blocks,
            dim=0,
        )
    target_rgb = (
        torch.cat(target_videos, dim=1)
        .permute(1, 0, 2, 3)
        .float()
        .div_(255.0)
        .unsqueeze(0)
    )
    return PreparedReconstructionBatch(
        history_latents=history_latents,
        history_c2w=history_c2w.unsqueeze(0),
        history_intrinsics=history_intrinsics.unsqueeze(0),
        target_rgb=target_rgb,
        target_c2w=target_c2w.unsqueeze(0),
        target_intrinsics=target_intrinsics.unsqueeze(0),
    )


class ReconstructionObjective(nn.Module):
    """Paper stage-one RGB MSE + LPIPS objective, without GAN/point maps."""

    def __init__(self, lpips_weight: float) -> None:
        super().__init__()
        self.lpips_weight = float(lpips_weight)
        if self.lpips_weight < 0:
            raise ValueError("lpips_weight cannot be negative")
        self.perceptual: nn.Module | None = None
        if self.lpips_weight > 0:
            try:
                import lpips
            except ImportError as error:
                raise ImportError(
                    "LPIPS supervision is enabled. Install the experiment "
                    "dependencies with `pip install -e '.[memory-reconstruction]'` "
                    "or set lpips_weight=0 for a pixel-MSE-only smoke test."
                ) from error
            self.perceptual = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target mismatch: {prediction.shape} vs {target.shape}"
            )
        prediction_float = prediction.float()
        target_float = target.float()
        mse = F.mse_loss(prediction_float, target_float)
        if self.perceptual is None:
            perceptual = mse.new_zeros(())
        else:
            batch, views, channels, height, width = prediction.shape
            prediction_2d = prediction_float.reshape(
                batch * views,
                channels,
                height,
                width,
            )
            target_2d = target_float.reshape_as(prediction_2d)
            # LPIPS expects RGB in [-1, 1]. Parameters stay frozen, but the
            # prediction path remains differentiable into decoder and memory.
            perceptual = self.perceptual(
                prediction_2d.mul(2.0).sub(1.0),
                target_2d.mul(2.0).sub(1.0),
            ).mean()
        loss = mse + self.lpips_weight * perceptual
        psnr = -10.0 * torch.log10(mse.detach().clamp_min(1e-12))
        return {
            "loss": loss,
            "rgb_mse": mse.detach(),
            "lpips": perceptual.detach(),
            "psnr": psnr,
        }


def reconstruction_training_step(
    model: GIMMemoryReconstructionModel,
    batch: PreparedReconstructionBatch,
    *,
    objective: ReconstructionObjective,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    target = batch.target_rgb.to(device=device, dtype=compute_dtype)
    prediction, memory = model(
        batch.history_latents.to(device=device, dtype=compute_dtype),
        batch.history_c2w.to(device),
        batch.history_intrinsics.to(device),
        batch.target_c2w.to(device),
        batch.target_intrinsics.to(device),
    )
    output = objective(prediction, target)
    output["memory_norm"] = memory.detach().float().norm(dim=-1).mean()
    output["prediction_mean"] = prediction.detach().float().mean()
    output["prediction_std"] = prediction.detach().float().std()
    return output
