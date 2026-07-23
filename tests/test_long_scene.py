from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from lingbot_video.long_scene.camera import (
    compute_world_to_scene_rotation,
    normalize_camera_poses,
    plucker_rays,
)
from lingbot_video.long_scene.config import LongSceneConfig
from lingbot_video.long_scene.data import (
    SyntheticLongSceneDataset,
    validate_scene_batch,
)
from lingbot_video.long_scene.losses import inverse_warp_features
from lingbot_video.long_scene.memory import MemorySource
from lingbot_video.long_scene.model import LongSceneWorldModel
from lingbot_video.long_scene.training import compute_long_scene_training_loss
from lingbot_video.transformer_lingbot_video import (
    LingBotVideoTransformer3DModel,
)


def tiny_config() -> LongSceneConfig:
    return LongSceneConfig(
        capture_chunk_size=3,
        slow_memory_tokens=8,
        fast_memory_tokens=4,
        memory_width=64,
        memory_heads=4,
        memory_update_layers=2,
        memory_consolidation_layers=2,
        memory_input_grid_h=4,
        memory_input_grid_w=4,
        consolidation_interval=2,
        truncate_memory_bptt_every=0,
        ray_fourier_bands=2,
        max_geometry_queries=32,
        max_memory_reconstruction_queries=32,
    )


def tiny_backbone() -> LingBotVideoTransformer3DModel:
    return LingBotVideoTransformer3DModel(
        patch_size=(1, 2, 2),
        in_channels=4,
        out_channels=4,
        hidden_size=64,
        num_attention_heads=4,
        depth=2,
        intermediate_size=128,
        text_dim=32,
        freq_dim=32,
        axes_dims=(4, 6, 6),
        axes_lens=(512, 128, 128),
        num_experts=0,
    )


def synthetic_batch(
    config: LongSceneConfig,
    *,
    capture_frames: int = 7,
) -> dict[str, torch.Tensor]:
    dataset = SyntheticLongSceneDataset(
        length=1,
        capture_frames=capture_frames,
        latent_channels=4,
        latent_frames=3,
        latent_height=8,
        latent_width=12,
        text_dim=32,
    )
    return next(iter(DataLoader(dataset, batch_size=1)))


def test_identity_center_ray_points_forward():
    c2w = torch.eye(4)[None, None]
    intrinsics = torch.tensor(
        [[[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]]
    )[None]
    ray = plucker_rays(c2w, intrinsics, 1, 1)[0, 0, 0, 0]
    torch.testing.assert_close(ray[:3], torch.zeros(3))
    torch.testing.assert_close(ray[3:6], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(ray[6:], torch.zeros(3))


def test_scene_frame_canonicalizes_reference_camera():
    cosine, sine = math.cos(0.7), math.sin(0.7)
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )
    pose[:3, 3] = torch.tensor([3.0, 1.0, -2.0])
    c2w = pose[None, None]
    world_to_scene = compute_world_to_scene_rotation(c2w)
    normalized = normalize_camera_poses(
        c2w,
        scene_center=pose[:3, 3][None],
        scene_scale=torch.ones(1),
        world_to_scene_rotation=world_to_scene,
    )
    torch.testing.assert_close(
        normalized[0, 0], torch.eye(4), atol=1e-6, rtol=1e-6
    )


def test_identity_reprojection_is_exact():
    source = torch.randn(1, 3, 6, 8)
    depth = torch.ones(1, 6, 8)
    pose = torch.eye(4)[None]
    intrinsics = torch.tensor(
        [[[0.8, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]]]
    )
    warped, valid = inverse_warp_features(
        source, depth, pose, pose, intrinsics, intrinsics
    )
    assert valid.all()
    torch.testing.assert_close(warped, source, atol=2e-5, rtol=2e-5)


def test_source_depth_rejects_occluded_reprojection():
    source = torch.randn(1, 3, 6, 8)
    target_depth = torch.ones(1, 6, 8)
    occluding_source_depth = torch.full((1, 6, 8), 0.5)
    pose = torch.eye(4)[None]
    intrinsics = torch.tensor(
        [[[0.8, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]]]
    )
    _, valid = inverse_warp_features(
        source,
        target_depth,
        pose,
        pose,
        intrinsics,
        intrinsics,
        source_depth=occluding_source_depth,
    )
    assert not valid.any()


def test_recurrent_memory_has_fixed_budget_for_longer_capture():
    torch.manual_seed(0)
    config = tiny_config()
    model = LongSceneWorldModel(tiny_backbone(), config)
    short = synthetic_batch(config, capture_frames=4)
    long = synthetic_batch(config, capture_frames=11)
    short_memory = model.encode_scene_memory(
        capture_latents=short["capture_latents"],
        capture_c2w=short["capture_c2w"],
        capture_intrinsics=short["capture_intrinsics"],
        capture_valid_mask=short["capture_valid_mask"],
    )
    long_memory = model.encode_scene_memory(
        capture_latents=long["capture_latents"],
        capture_c2w=long["capture_c2w"],
        capture_intrinsics=long["capture_intrinsics"],
        capture_valid_mask=long["capture_valid_mask"],
    )
    expected = config.slow_memory_tokens + config.fast_memory_tokens
    assert short_memory.scene_tokens.shape == (1, expected, config.memory_width)
    assert long_memory.scene_tokens.shape == (1, expected, config.memory_width)
    assert short_memory.state.steps.item() == 4
    assert long_memory.state.steps.item() == 11


def test_generated_write_is_immediate_and_branch_local():
    torch.manual_seed(1)
    config = tiny_config()
    model = LongSceneWorldModel(tiny_backbone(), config)
    batch = synthetic_batch(config)
    base = model.encode_scene_memory(
        capture_latents=batch["capture_latents"],
        capture_c2w=batch["capture_c2w"],
        capture_intrinsics=batch["capture_intrinsics"],
        capture_valid_mask=batch["capture_valid_mask"],
    )
    observations = batch["target_latents"].permute(0, 2, 1, 3, 4)
    updated, diagnostics = model.update_scene_memory(
        base,
        latents=observations,
        c2w=batch["target_c2w"],
        intrinsics=batch["target_intrinsics"],
        valid_mask=batch["target_write_valid_mask"],
        source_type=MemorySource.GENERATED,
        consolidate=False,
    )
    assert not base.state.fast_confidence.bool().any()
    assert updated.state.fast_confidence.bool().all()
    assert diagnostics.slow_write_gate is None
    torch.testing.assert_close(
        base.state.slow_tokens, updated.state.slow_tokens
    )
    assert not torch.equal(base.state.fast_tokens, updated.state.fast_tokens)


def test_verified_generated_evidence_can_consolidate():
    torch.manual_seed(2)
    config = tiny_config()
    model = LongSceneWorldModel(tiny_backbone(), config)
    batch = synthetic_batch(config)
    base = model.encode_scene_memory(
        capture_latents=batch["capture_latents"],
        capture_c2w=batch["capture_c2w"],
        capture_intrinsics=batch["capture_intrinsics"],
        capture_valid_mask=batch["capture_valid_mask"],
    )
    observations = batch["target_latents"].permute(0, 2, 1, 3, 4)
    verified, diagnostics = model.update_scene_memory(
        base,
        latents=observations,
        c2w=batch["target_c2w"],
        intrinsics=batch["target_intrinsics"],
        valid_mask=batch["target_write_valid_mask"],
        source_type=MemorySource.VERIFIED_GENERATED,
        consolidate=True,
    )
    assert diagnostics.slow_write_gate is not None
    assert not torch.equal(base.state.slow_tokens, verified.state.slow_tokens)
    assert not verified.state.fast_confidence.bool().any()


def test_tiny_end_to_end_rollout_training_step_has_gradients():
    torch.manual_seed(3)
    config = tiny_config()
    backbone = tiny_backbone()
    backbone.enable_gradient_checkpointing()
    backbone.requires_grad_(False)
    model = LongSceneWorldModel(backbone, config)
    batch = synthetic_batch(config)
    validate_scene_batch(batch)
    output = compute_long_scene_training_loss(model, batch, config)
    assert torch.isfinite(output.loss)
    assert "rollout_flow" in output.losses
    assert "memory_state_consistency" in output.losses
    assert "memory_reconstruction" in output.losses
    output.loss.backward()
    assert model.camera_encoder.output_gate.grad is not None
    assert model.scene_memory.condition_gate.grad is not None
    assert model.scene_memory.initial_slow.grad is not None
    assert model.scene_memory.fast_writer.write_gate[-1].weight.grad is not None
    assert model.geometry_head.depth_head[-1].weight.grad is not None
    assert model.memory_readout_head.feature_head[-1].weight.grad is not None


def test_backbone_camera_bias_is_backward_compatible():
    torch.manual_seed(4)
    backbone = tiny_backbone().eval()
    hidden = torch.randn(1, 4, 3, 8, 12)
    prompt = torch.randn(1, 5, 32)
    mask = torch.ones(1, 5, dtype=torch.long)
    timestep = torch.tensor([500.0])
    with torch.no_grad():
        baseline = backbone(
            hidden,
            timestep,
            prompt,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
        zero_bias = torch.zeros(1, 3 * 4 * 6, 64)
        with_bias_argument = backbone(
            hidden,
            timestep,
            prompt,
            encoder_attention_mask=mask,
            video_token_bias=zero_bias,
            return_dict=False,
        )[0]
    torch.testing.assert_close(baseline, with_bias_argument)
