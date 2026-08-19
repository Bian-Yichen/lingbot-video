from __future__ import annotations

import torch
import torch.nn as nn

from lingbot_video.geometry_aware_memory.wan_latent_reconstruction import (
    GIMWanLatentReconstructionModel,
    HistoryViewCurriculum,
    LoRAConv2d1x1,
    WanDecoderBridge,
    WanLatentReconstructionConfig,
)
from lingbot_video.geometry_aware_memory.wan_latent_training import (
    WanLatentReconstructionObjective,
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


class _DummyWanAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_qkv = nn.Conv2d(4, 12, 1)
        self.proj = nn.Conv2d(4, 4, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query, key, value = self.to_qkv(hidden_states).chunk(3, dim=1)
        return self.proj(torch.tanh(query + key + value))


class _DummyWanMidBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([_DummyWanAttention()])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.attentions[0](hidden_states)


class _DummyWanDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mid_block = _DummyWanMidBlock()
        self.output = nn.Conv2d(4, 3, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool,
    ) -> torch.Tensor:
        assert feat_cache is None
        assert feat_idx is None
        assert first_chunk
        decoded = self.output(self.mid_block(hidden_states.squeeze(2)))
        return decoded.mul(0.1).unsqueeze(2)


def _model() -> GIMWanLatentReconstructionModel:
    bridge = WanDecoderBridge(
        nn.Conv3d(4, 4, 1),
        _DummyWanDecoder(),
        latent_channels=4,
        refiner_hidden_size=8,
        lora_rank=2,
        lora_alpha=2,
    )
    config = WanLatentReconstructionConfig(
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
        renderer_hidden_size=32,
        renderer_depth=2,
        renderer_num_heads=4,
        decoder_lora_rank=2,
        decoder_lora_alpha=2,
        decoder_refiner_hidden_size=8,
        vae_latents_mean=(0.0, 0.0, 0.0, 0.0),
        vae_latents_std=(1.0, 1.0, 1.0, 1.0),
    )
    return GIMWanLatentReconstructionModel(
        nn.Linear(16, 32),
        bridge,
        config,
    )


def _forward(
    model: GIMWanLatentReconstructionModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_c2w, history_intrinsics = _cameras(1, 3)
    target_c2w, target_intrinsics = _cameras(1, 2)
    return model(
        torch.randn(1, 4, 3, 4, 4),
        history_c2w,
        history_intrinsics,
        target_c2w,
        target_intrinsics,
    )


def test_history_curriculum_reaches_full_range_and_never_decreases() -> None:
    curriculum = HistoryViewCurriculum(
        start_min=2,
        start_max=4,
        end_min=12,
        end_max=32,
        curriculum_epochs=8,
    )
    values = [curriculum.views_for_epoch(epoch) for epoch in range(10)]
    assert values == [
        (2, 4),
        (3, 8),
        (5, 12),
        (6, 16),
        (8, 20),
        (9, 24),
        (11, 28),
        (12, 32),
        (12, 32),
        (12, 32),
    ]


def test_renderer_predicts_multiple_independent_wan_latents() -> None:
    model = _model()
    predicted_latents, predicted_rgb, memory = _forward(model)
    assert predicted_latents.shape == (1, 2, 4, 4, 4)
    assert predicted_rgb.shape == (1, 2, 3, 4, 4)
    assert memory.shape == (1, 4, 32)
    assert torch.all((predicted_rgb >= 0) & (predicted_rgb <= 1))


def test_stage_one_trains_memory_renderer_but_gates_decoder_adapters() -> None:
    model = _model()
    model.set_training_stage(1)
    model.wan_decoder.gradient_checkpointing = True
    predicted_latents, predicted_rgb, _ = _forward(model)
    loss = predicted_latents.square().mean() + predicted_rgb.square().mean()
    loss.backward()
    assert model.patch_embedder.weight.grad is None
    assert model.memory_encoder.memory_queries.grad is not None
    assert model.latent_renderer.output_projection.weight.grad is not None
    lora_modules = [
        module for module in model.modules() if isinstance(module, LoRAConv2d1x1)
    ]
    assert lora_modules
    assert all(module.up.weight.grad is not None for module in lora_modules)
    assert all(
        torch.count_nonzero(module.up.weight.grad) == 0
        for module in lora_modules
    )
    assert all(module.base.weight.grad is None for module in lora_modules)


def test_stage_two_trains_decoder_lora_and_zero_initialized_refiner() -> None:
    model = _model()
    model.set_training_stage(2)
    predicted_latents, predicted_rgb, _ = _forward(model)
    loss = predicted_latents.square().mean() + predicted_rgb.square().mean()
    loss.backward()
    lora_modules = [
        module for module in model.modules() if isinstance(module, LoRAConv2d1x1)
    ]
    assert any(
        torch.count_nonzero(module.up.weight.grad) > 0
        for module in lora_modules
    )
    refiner_output = model.wan_decoder.latent_refiner.net[-1]
    assert torch.count_nonzero(refiner_output.weight.grad) > 0
    assert all(module.base.weight.grad is None for module in lora_modules)


def test_checkpoint_excludes_frozen_wan_decoder_base() -> None:
    model = _model()
    state = model.experiment_state_dict()
    assert any(name.startswith("memory_encoder.") for name in state)
    assert any(name.startswith("latent_renderer.") for name in state)
    assert any(".up.weight" in name for name in state)
    assert not any(".base." in name for name in state)
    assert not any(name.startswith("wan_decoder.post_quant_conv") for name in state)


def test_joint_objective_supervises_latent_and_rgb() -> None:
    objective = WanLatentReconstructionObjective(
        latent_weight=1.0,
        rgb_weight=1.0,
        lpips_weight=0.0,
    )
    predicted_latents = torch.zeros(1, 2, 4, 4, 4, requires_grad=True)
    predicted_rgb = torch.zeros(1, 2, 3, 4, 4, requires_grad=True)
    output = objective(
        predicted_latents,
        torch.ones_like(predicted_latents),
        predicted_rgb,
        torch.ones_like(predicted_rgb),
    )
    assert output["loss"].item() == 2.0
    output["loss"].backward()
    assert torch.count_nonzero(predicted_latents.grad) > 0
    assert torch.count_nonzero(predicted_rgb.grad) > 0
