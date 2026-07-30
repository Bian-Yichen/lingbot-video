from __future__ import annotations

import pytest
import torch

import lingbot_video.transformer_lingbot_video as lingbot_transformer
from lingbot_video.geometry_aware_memory.data import (
    GeometryMemorySampleConfig,
    LocalRoomTourIndex,
)
from lingbot_video.geometry_aware_memory.memory_encoder import (
    GIMImplicitMemoryEncoder,
    GIMMemoryEncoderConfig,
)
from lingbot_video.geometry_aware_memory.inference import DynamicGIMHistory
from lingbot_video.geometry_aware_memory.geometry import (
    make_origin_direction_rays,
)
from lingbot_video.geometry_aware_memory.pruning import (
    MIGreedyPruner,
    PoseTimeKernelConfig,
)
from lingbot_video.geometry_aware_memory.teacher import vggt_target_hw
from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel


def test_vggt_grid_matches_roomtour_default() -> None:
    assert vggt_target_hw((480, 832)) == (294, 518)


def test_local_roomtour_index_uses_mounted_scene_in_place(tmp_path) -> None:
    scene = tmp_path / "scene_000.mp4"
    scene.mkdir()
    index = LocalRoomTourIndex(tmp_path)
    assert index.list_items() == ["scene_000.mp4"]
    assert index.item_path("scene_000.mp4") == scene
    with pytest.raises(ValueError, match="direct child"):
        index.item_path("../scene_000.mp4")


def test_offline_context_rejects_target_leaking_guard() -> None:
    with pytest.raises(ValueError, match="target_guard_rgb_frames >= 128"):
        GeometryMemorySampleConfig(
            context_policy="all_except_target",
            target_guard_rgb_frames=0,
        ).validate()


def test_ray_at_principal_point_is_camera_forward() -> None:
    c2w = torch.eye(4).reshape(1, 4, 4)
    intrinsics = torch.tensor(
        [[[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]]
    )
    rays = make_origin_direction_rays(c2w, intrinsics, 3, 3)
    torch.testing.assert_close(
        rays[0, 1, 1],
        torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_mi_pruner_returns_unique_subset() -> None:
    c2w = torch.eye(4).repeat(12, 1, 1)
    c2w[:, 0, 3] = torch.linspace(0, 11, 12)
    times = torch.arange(12).float()
    selected = MIGreedyPruner(
        PoseTimeKernelConfig(
            sigma_position=2.0,
            sigma_rotation=0.5,
            sigma_time=3.0,
        )
    ).select(c2w, times, budget=5)
    assert selected.shape == (5,)
    assert selected.unique().numel() == 5
    assert int(selected.min()) >= 0
    assert int(selected.max()) < 12


def test_memory_encoder_fixed_output_shape_and_gradients() -> None:
    config = GIMMemoryEncoderConfig(
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        memory_latent_frames=3,
        patch_height=4,
        patch_width=6,
        compact_stride=2,
        depth=2,
        camera_input_dim=16,
        axes_dims=(2, 2, 4),
        axes_lens=(32, 32, 32),
    )
    encoder = GIMImplicitMemoryEncoder(config)
    history = torch.randn(2, 5, 24, 32, requires_grad=True)
    cameras = torch.randn(2, 5, 16)
    memory = encoder(history, cameras)
    assert memory.shape == (2, 3 * 4 * 6, 32)
    memory.square().mean().backward()
    assert history.grad is not None
    assert encoder.memory_queries.grad is not None


def test_lingbot_memory_is_temporal_prefix_but_output_is_target_only() -> None:
    model = LingBotVideoTransformer3DModel(
        patch_size=(1, 2, 2),
        in_channels=4,
        out_channels=4,
        hidden_size=32,
        num_attention_heads=4,
        depth=2,
        intermediate_size=64,
        text_dim=16,
        freq_dim=16,
        axes_dims=(2, 2, 4),
        axes_lens=(128, 32, 32),
    )
    target = torch.randn(1, 4, 3, 8, 12)
    text = torch.randn(1, 5, 16)
    memory = torch.randn(1, 2 * 4 * 6, 32)
    actions = torch.randn(1, 3, 32)
    output = model(
        target,
        torch.tensor([500.0]),
        text,
        memory_hidden_states=memory,
        video_action_embeds=actions,
        return_dict=False,
    )[0]
    assert output.shape == target.shape


def test_packed_batch_uses_sdpa_without_flashattention3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lingbot_transformer,
        "flash_attn_varlen_func_v3",
        None,
    )
    model = LingBotVideoTransformer3DModel(
        patch_size=(1, 2, 2),
        in_channels=4,
        out_channels=4,
        hidden_size=32,
        num_attention_heads=4,
        depth=1,
        intermediate_size=64,
        text_dim=16,
        freq_dim=16,
        axes_dims=(2, 2, 4),
        axes_lens=(128, 32, 32),
    )
    target = torch.randn(2, 4, 3, 8, 12)
    text = torch.randn(2, 5, 16)
    memory = torch.randn(2, 2 * 4 * 6, 32)
    actions = torch.randn(2, 3, 32)
    output = model(
        target,
        torch.tensor([500.0, 250.0]),
        text,
        memory_hidden_states=memory,
        video_action_embeds=actions,
        return_dict=False,
    )[0]
    assert output.shape == target.shape


def test_dynamic_history_appends_generated_latents_and_cameras() -> None:
    history = DynamicGIMHistory(
        latents=torch.randn(1, 4, 3, 2, 2),
        c2w=torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1),
        intrinsics=torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 3, 1, 1),
        times=torch.tensor([0, 4, 8]),
    )
    history.append(
        torch.randn(1, 4, 2, 2, 2),
        torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1),
        torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
        torch.tensor([12, 16]),
    )
    assert history.frame_count == 5
    assert history.times.tolist() == [0, 4, 8, 12, 16]
