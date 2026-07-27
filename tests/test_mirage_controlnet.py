import torch

from lingbot_video.latent_spatial_memory.controlnet import (
    LingBotLatentMemoryControlNet,
)
from lingbot_video.latent_spatial_memory.model import (
    LatentMetricDepthHead,
    LingBotVideoLatentMemoryModel,
)
from lingbot_video.transformer_lingbot_video import (
    LingBotVideoTransformer3DModel,
)


def _tiny_backbone() -> LingBotVideoTransformer3DModel:
    return LingBotVideoTransformer3DModel(
        patch_size=(1, 2, 2),
        in_channels=4,
        out_channels=4,
        hidden_size=24,
        num_attention_heads=2,
        depth=4,
        intermediate_size=48,
        text_dim=8,
        freq_dim=16,
        axes_dims=(4, 4, 4),
        axes_lens=(64, 16, 16),
    )


def test_mirage_forward_uses_reference_target_preceding_layout() -> None:
    torch.manual_seed(5)
    backbone = _tiny_backbone()
    controlnet = LingBotLatentMemoryControlNet.from_backbone(
        backbone,
        (0, 2),
    )
    model = LingBotVideoLatentMemoryModel(
        backbone,
        controlnet,
        LatentMetricDepthHead(4, hidden_channels=8),
    )
    batch, channels, target, preceding, reference, height, width = (
        1,
        4,
        3,
        2,
        1,
        4,
        4,
    )
    output = model(
        noisy_latents=torch.randn(batch, channels, target, height, width),
        target_timesteps=torch.tensor([[0.0, 400.0, 700.0]]),
        encoder_hidden_states=torch.randn(batch, 3, 8),
        encoder_attention_mask=torch.ones(batch, 3, dtype=torch.long),
        memory_latents=torch.randn(
            batch,
            channels,
            target + preceding,
            height,
            width,
        ),
        memory_visibility=torch.ones(
            batch,
            1,
            target + preceding,
            height,
            width,
        ),
        target_rays=torch.randn(batch, 6, target, height, width),
        preceding_latents=torch.randn(
            batch,
            channels,
            preceding,
            height,
            width,
        ),
        preceding_rays=torch.randn(batch, 6, preceding, height, width),
        reference_latents=torch.randn(
            batch,
            channels,
            reference,
            height,
            width,
        ),
    )

    assert output.velocity.shape == (
        batch,
        channels,
        target,
        height,
        width,
    )
    expected_control_tokens = (target + preceding) * (height // 2) * (width // 2)
    assert set(output.control_residuals) == {0, 2}
    for residual in output.control_residuals.values():
        assert residual.shape == (batch, expected_control_tokens, 24)
        torch.testing.assert_close(residual, torch.zeros_like(residual))

