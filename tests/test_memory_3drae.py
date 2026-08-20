from __future__ import annotations

import json

import torch
import torch.nn as nn
from PIL import Image

from lingbot_video.geometry_aware_memory.three_drae import (
    ThreeDRAEConfig,
    ThreeDRAEViewCurriculum,
    WanThreeDRAEModel,
)
from lingbot_video.geometry_aware_memory.three_drae_training import (
    ThreeDRAEObjective,
    adaptive_adversarial_weight,
    hinge_discriminator_loss,
    save_three_drae_visualization,
)
from lingbot_video.geometry_aware_memory.wan_latent_reconstruction import (
    LoRAConv2d1x1,
    WanDecoderBridge,
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
        first_chunk: bool,
    ) -> torch.Tensor:
        assert first_chunk
        decoded = self.output(self.mid_block(hidden_states.squeeze(2)))
        return decoded.mul(0.1).unsqueeze(2)


def _model(*, decoder_noise_tau: float = 0.0) -> WanThreeDRAEModel:
    bridge = WanDecoderBridge(
        nn.Conv3d(4, 4, 1),
        _DummyWanDecoder(),
        latent_channels=4,
        refiner_hidden_size=8,
        lora_rank=2,
        lora_alpha=2,
    )
    config = ThreeDRAEConfig(
        image_height=8,
        image_width=8,
        vae_spatial_stride=2,
        latent_channels=4,
        latent_patch_size=(1, 2, 2),
        hidden_size=32,
        num_heads=4,
        num_memory_tokens=6,
        encoder_depth=2,
        decoder_depth=2,
        latent_batch_norm=False,
        decoder_noise_tau=decoder_noise_tau,
        view_mask_probability=0.0,
        decoder_lora_rank=2,
        decoder_lora_alpha=2,
        decoder_refiner_hidden_size=8,
        vae_latents_mean=(0.0, 0.0, 0.0, 0.0),
        vae_latents_std=(1.0, 1.0, 1.0, 1.0),
    )
    return WanThreeDRAEModel(bridge, config)


def _forward(
    model: WanThreeDRAEModel,
    *,
    visibility: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    history_c2w, history_intrinsics = _cameras(1, 3)
    target_c2w, target_intrinsics = _cameras(1, 2)
    return model(
        torch.randn(1, 4, 3, 4, 4),
        history_c2w,
        history_intrinsics,
        target_c2w,
        target_intrinsics,
        view_visibility=visibility,
    )


def test_history_and_query_curriculum_increase_together() -> None:
    curriculum = ThreeDRAEViewCurriculum(
        history_start_min=2,
        history_start_max=4,
        history_end_min=12,
        history_end_max=32,
        query_start=2,
        query_end=8,
        curriculum_epochs=8,
    )
    values = [curriculum.values_for_epoch(epoch) for epoch in range(10)]
    assert values == [
        (2, 4, 2),
        (3, 8, 3),
        (5, 12, 4),
        (6, 16, 5),
        (8, 20, 5),
        (9, 24, 6),
        (11, 28, 7),
        (12, 32, 8),
        (12, 32, 8),
        (12, 32, 8),
    ]


def test_three_drae_has_fixed_memory_and_multiple_novel_views() -> None:
    model = _model().eval()
    predicted_latents, predicted_rgb, memory, visibility = _forward(model)
    assert predicted_latents.shape == (1, 2, 4, 4, 4)
    assert predicted_rgb.shape == (1, 2, 3, 4, 4)
    assert memory.shape == (1, 6, 32)
    assert visibility.shape == (1, 3)
    assert torch.all(visibility)
    assert len(model.memory_encoder.blocks) == 2
    assert len(model.latent_decoder.blocks) == 2


def test_invisible_history_views_are_explicit_model_inputs() -> None:
    model = _model().eval()
    requested = torch.tensor([[True, False, True]])
    _, _, _, returned = _forward(model, visibility=requested)
    assert torch.equal(returned, requested)


def test_stage_one_trains_3drae_but_not_frozen_wan_base() -> None:
    model = _model()
    model.set_training_stage(1)
    predicted_latents, predicted_rgb, _, _ = _forward(model)
    (predicted_latents.square().mean() + predicted_rgb.square().mean()).backward()
    assert model.memory_encoder.memory_queries.grad is not None
    assert model.latent_decoder.output_projection.weight.grad is not None
    lora_modules = [
        module for module in model.modules() if isinstance(module, LoRAConv2d1x1)
    ]
    assert lora_modules
    assert all(module.base.weight.grad is None for module in lora_modules)
    assert all(
        module.up.weight.grad is not None
        and torch.count_nonzero(module.up.weight.grad) == 0
        for module in lora_modules
    )


def test_checkpoint_contains_3drae_but_excludes_wan_base() -> None:
    state = _model().experiment_state_dict()
    assert any(name.startswith("input_projection.") for name in state)
    assert any(name.startswith("ray_embedding.") for name in state)
    assert any(name.startswith("memory_encoder.") for name in state)
    assert any(name.startswith("latent_decoder.") for name in state)
    assert any(".up.weight" in name for name in state)
    assert not any(".base." in name for name in state)
    assert not any(name.startswith("wan_decoder.post_quant_conv") for name in state)


def test_paper_reconstruction_and_gan_helpers_are_differentiable() -> None:
    objective = ThreeDRAEObjective(
        latent_weight=0.0,
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
    assert output["loss"].item() == 1.0
    assert hinge_discriminator_loss(
        torch.ones(2),
        -torch.ones(2),
    ).item() == 0.0

    projection = nn.Linear(4, 1, bias=False)
    features = torch.ones(2, 4)
    prediction = projection(features)
    reconstruction = prediction.square().mean()
    adversarial = -prediction.mean()
    weight = adaptive_adversarial_weight(
        reconstruction,
        adversarial,
        projection.weight,
    )
    assert weight.ndim == 0
    assert torch.isfinite(weight)


def test_training_visualization_saves_prediction_target_and_metadata(
    tmp_path,
) -> None:
    predicted = torch.zeros(1, 2, 3, 4, 6)
    target = torch.ones_like(predicted)
    image_dir = save_three_drae_visualization(
        tmp_path,
        item_name="scene/example.mp4",
        history_indices=[1, 2, 8],
        target_indices=[4, 5],
        predicted_rgb=predicted,
        target_rgb=target,
        metrics={"rgb_mse": 1.0},
        global_iteration=200,
        global_step=100,
        epoch=1,
        stage=1,
    )
    assert image_dir == (
        tmp_path
        / "images"
        / "iter-00000200-step-00000100"
        / "scene_example.mp4"
    )
    assert len(list(image_dir.glob("*-prediction.png"))) == 2
    assert len(list(image_dir.glob("*-target.png"))) == 2
    comparisons = list(image_dir.glob("*-comparison.png"))
    assert len(comparisons) == 2
    with Image.open(comparisons[0]) as image:
        assert image.size == (12, 4)
    metadata = json.loads((image_dir / "metadata.json").read_text())
    assert metadata["comparison_layout"] == "prediction_left_target_right"
    assert metadata["target_indices_internal"] == [4, 5]
