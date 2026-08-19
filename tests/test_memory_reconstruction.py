from __future__ import annotations

import torch

from lingbot_video.geometry_aware_memory.memory_reconstruction import (
    GIMMemoryReconstructionModel,
    MemoryReconstructionConfig,
)
from lingbot_video.geometry_aware_memory.reconstruction_training import (
    ReconstructionObjective,
)


def _cameras(batch: int, views: int) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, views, 1, 1)
    c2w[:, :, 0, 3] = torch.linspace(0.0, 0.2, views)
    intrinsics = torch.zeros(batch, views, 3, 3)
    intrinsics[..., 0, 0] = 4.0
    intrinsics[..., 1, 1] = 4.0
    intrinsics[..., 0, 2] = 3.5
    intrinsics[..., 1, 2] = 3.5
    intrinsics[..., 2, 2] = 1.0
    return c2w, intrinsics


def _model() -> GIMMemoryReconstructionModel:
    config = MemoryReconstructionConfig(
        image_height=8,
        image_width=8,
        vae_spatial_stride=2,
        latent_channels=4,
        latent_patch_size=(1, 2, 2),
        encoder_hidden_size=32,
        encoder_num_heads=4,
        encoder_axes_dims=(2, 2, 4),
        encoder_axes_lens=(32, 32, 32),
        memory_latent_frames=1,
        compact_stride=1,
        decoder_hidden_size=32,
        decoder_depth=2,
        decoder_num_heads=4,
    )
    return GIMMemoryReconstructionModel(torch.nn.Linear(16, 32), config)


def test_memory_decoder_reconstructs_multiple_novel_views() -> None:
    model = _model()
    history_latents = torch.randn(1, 4, 3, 4, 4)
    history_c2w, history_intrinsics = _cameras(1, 3)
    target_c2w, target_intrinsics = _cameras(1, 2)
    prediction, memory = model(
        history_latents,
        history_c2w,
        history_intrinsics,
        target_c2w,
        target_intrinsics,
    )
    assert prediction.shape == (1, 2, 3, 8, 8)
    assert memory.shape == (1, 4, 32)
    assert torch.all((prediction >= 0) & (prediction <= 1))


def test_reconstruction_loss_trains_memory_and_decoder_but_not_patch_embedder() -> None:
    model = _model()
    objective = ReconstructionObjective(lpips_weight=0.0)
    history_c2w, history_intrinsics = _cameras(1, 2)
    target_c2w, target_intrinsics = _cameras(1, 1)
    prediction, _ = model(
        torch.randn(1, 4, 2, 4, 4),
        history_c2w,
        history_intrinsics,
        target_c2w,
        target_intrinsics,
    )
    loss = objective(prediction, torch.rand_like(prediction))["loss"]
    loss.backward()
    assert model.patch_embedder.weight.grad is None
    assert model.memory_encoder.memory_queries.grad is not None
    assert model.decoder.output_projection.weight.grad is not None
