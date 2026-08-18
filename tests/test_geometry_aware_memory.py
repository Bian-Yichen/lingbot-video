from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lingbot_video.transformer_lingbot_video as lingbot_transformer
from lingbot_video.geometry_aware_memory.data import (
    estimate_sparse_frame_count,
    GeometryMemorySampleConfig,
    has_local_retrieval_sample_for_frame_count,
    LocalRoomTourIndex,
    LocalVipeRoomTourDataset,
    VipeRoomTourItem,
)
from lingbot_video.geometry_aware_memory.memory_encoder import (
    CompactLinear,
    CompactSelfAttention,
    GIMImplicitMemoryEncoder,
    GIMMemoryEncoderConfig,
)
from lingbot_video.geometry_aware_memory.lora import (
    BackboneLoRAConfig,
    LoRALinear,
    inject_backbone_lora,
)
from lingbot_video.geometry_aware_memory.inference import DynamicGIMHistory
from lingbot_video.geometry_aware_memory.model import (
    GIMWorldLingBotModel,
    GIMWorldModelConfig,
)
from lingbot_video.geometry_aware_memory.geometry import (
    make_origin_direction_rays,
)
from lingbot_video.geometry_aware_memory.pruning import (
    MIGreedyPruner,
    PoseTimeKernelConfig,
)
from lingbot_video.geometry_aware_memory.teacher import vggt_target_hw
from lingbot_video.geometry_aware_memory.training import (
    _require_independent_wan_vae,
    encode_wan_frames_independently,
)
from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
from scripts.inference_geometry_aware_memory import (
    _sample_epoch_from_checkpoint,
    _validate_loaded_state,
)


def test_vggt_grid_matches_roomtour_default() -> None:
    assert vggt_target_hw((480, 832)) == (294, 518)


def test_inference_defaults_to_checkpoint_last_completed_epoch() -> None:
    assert _sample_epoch_from_checkpoint({"next_epoch": 1}, None) == 0
    assert _sample_epoch_from_checkpoint({"next_epoch": 5}, None) == 4
    assert _sample_epoch_from_checkpoint({"next_epoch": 5}, 2) == 2
    with pytest.raises(ValueError, match="cannot be negative"):
        _sample_epoch_from_checkpoint({"next_epoch": 5}, -1)


def test_inference_checkpoint_validation_allows_partial_backbones() -> None:
    _validate_loaded_state([], [], backbone_train_mode="full")
    _validate_loaded_state(
        ["backbone.blocks.0.weight"],
        [],
        backbone_train_mode="frozen",
    )
    _validate_loaded_state(
        ["backbone.blocks.0.attn.to_q.base_layer.weight"],
        [],
        backbone_train_mode="lora",
    )
    with pytest.raises(RuntimeError, match="lora_a"):
        _validate_loaded_state(
            ["backbone.blocks.0.attn.to_q.lora_a.weight"],
            [],
            backbone_train_mode="lora",
        )
    with pytest.raises(RuntimeError, match="memory_encoder"):
        _validate_loaded_state(
            ["memory_encoder.memory_queries"],
            [],
            backbone_train_mode="frozen",
        )
    with pytest.raises(RuntimeError, match="unexpected"):
        _validate_loaded_state(
            [],
            ["unknown.weight"],
            backbone_train_mode="full",
        )


def test_local_roomtour_index_uses_mounted_scene_in_place(tmp_path) -> None:
    scene = tmp_path / "scene_000.mp4"
    scene.mkdir()
    index = LocalRoomTourIndex(tmp_path)
    assert index.list_items() == ["scene_000.mp4"]
    assert index.item_path("scene_000.mp4") == scene
    with pytest.raises(ValueError, match="direct child"):
        index.item_path("../scene_000.mp4")


def test_filename_frame_count_and_epoch_scene_filtering(tmp_path) -> None:
    tiny_name = "tiny_000000_000300.mp4"
    short_name = "qv2GbEEaivI_000000_001182.mp4"
    medium_name = "medium_000000_000405.mp4"
    long_name = "long_000000_005000.mp4"
    for item_name in (tiny_name, short_name, medium_name, long_name):
        (tmp_path / item_name).mkdir()
    assert estimate_sparse_frame_count(tiny_name) == 60
    assert estimate_sparse_frame_count(short_name) == 236
    assert estimate_sparse_frame_count(medium_name) == 81
    assert estimate_sparse_frame_count(long_name) == 1000

    config = GeometryMemorySampleConfig(
        target_rgb_frames=41,
        local_window_rgb_frames=81,
        memory_views_min=2,
        memory_views_max=24,
    )
    dataset = LocalVipeRoomTourDataset(
        tmp_path,
        config,
        item_list=[tiny_name, short_name, medium_name, long_name],
    )
    assert dataset.total_item_count == 4
    assert dataset.items == [short_name, medium_name, long_name]
    assert dataset.skipped_items == (tiny_name,)

    dataset.set_epoch(2)
    assert dataset.items == [short_name, medium_name, long_name]
    assert dataset.skipped_items == (tiny_name,)


def test_frame_count_filter_includes_exact_valid_boundary() -> None:
    config = GeometryMemorySampleConfig(
        target_rgb_frames=41,
        local_window_rgb_frames=81,
    )
    assert not has_local_retrieval_sample_for_frame_count(80, config)
    assert has_local_retrieval_sample_for_frame_count(81, config)


def test_local_retrieval_config_requires_non_target_candidates() -> None:
    with pytest.raises(ValueError, match="local_window_rgb_frames"):
        GeometryMemorySampleConfig(
            target_rgb_frames=41,
            local_window_rgb_frames=42,
        ).validate()
    with pytest.raises(ValueError, match="memory_views_max"):
        GeometryMemorySampleConfig(
            target_rgb_frames=41,
            local_window_rgb_frames=50,
            memory_views_max=10,
        ).validate()


def test_independent_wan_vae_validation_does_not_require_causal_state() -> None:
    class IndependentWanVAE:
        config = SimpleNamespace(
            scale_factor_temporal=4,
            patch_size=None,
        )
        encode = object()

    _require_independent_wan_vae(IndependentWanVAE())


def test_independent_wan_encoding_preserves_one_latent_per_frame() -> None:
    class FakeDistribution:
        def __init__(self, value: torch.Tensor) -> None:
            self.value = value

        def mode(self) -> torch.Tensor:
            return self.value

    class FakeWanVAE:
        config = SimpleNamespace(
            patch_size=None,
            latents_mean=[0.0, 0.0],
            latents_std=[1.0, 1.0],
        )

        def encode(self, images: torch.Tensor) -> SimpleNamespace:
            latents = torch.nn.functional.avg_pool3d(
                images[:, :2],
                kernel_size=(1, 4, 4),
            )
            return SimpleNamespace(
                latent_dist=FakeDistribution(latents)
            )

    class FakeItem:
        def read_video(
            self,
            indices: list[int],
            target_hw: tuple[int, int],
        ) -> torch.Tensor:
            del target_hw
            return torch.stack(
                [
                    torch.full((3, 8, 8), float(index) / 10.0)
                    for index in indices
                ],
                dim=1,
            )

    output = encode_wan_frames_independently(
        FakeWanVAE(),
        FakeItem(),
        [0, 1, 2, 3, 4],
        (8, 8),
        read_chunk_rgb_frames=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert output.shape == (1, 2, 5, 2, 2)
    assert torch.all(output[:, :, 1] > output[:, :, 0])
    assert torch.all(output[:, :, 4] > output[:, :, 3])


def test_independent_wan_encoding_accepts_worker_preloaded_uint8() -> None:
    class FakeDistribution:
        def __init__(self, value: torch.Tensor) -> None:
            self.value = value

        def mode(self) -> torch.Tensor:
            return self.value

    class FakeWanVAE:
        config = SimpleNamespace(
            patch_size=None,
            latents_mean=[0.0, 0.0],
            latents_std=[1.0, 1.0],
        )

        def encode(self, images: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(
                latent_dist=FakeDistribution(
                    torch.nn.functional.avg_pool3d(
                        images[:, :2],
                        kernel_size=(1, 4, 4),
                    )
                )
            )

    video = torch.stack(
        [torch.full((3, 8, 8), index * 32, dtype=torch.uint8)
         for index in range(5)],
        dim=1,
    )
    output = encode_wan_frames_independently(
        FakeWanVAE(),
        None,
        [0, 1, 2, 3, 4],
        (8, 8),
        read_chunk_rgb_frames=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        preloaded_video=video,
    )
    assert output.shape == (1, 2, 5, 2, 2)
    assert torch.all(output[:, :, 1] > output[:, :, 0])
    assert torch.all(output[:, :, 4] > output[:, :, 3])


def test_sample_has_continuous_target_and_pose_retrieved_small_memory() -> None:
    item = VipeRoomTourItem.__new__(VipeRoomTourItem)
    item.root = Path("/fake/scene.mp4")
    item.indices = list(range(1000))
    item.pose_by_index = {}
    for index in item.indices:
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = np.sin(index / 80.0)
        pose[1, 3] = np.cos(index / 80.0)
        pose[2, 3] = index / 1000.0
        item.pose_by_index[index] = pose
    config = GeometryMemorySampleConfig(
        target_rgb_frames=41,
        query_blocks=1,
        local_window_rgb_frames=81,
        memory_views_min=2,
        memory_views_max=24,
    )
    sample = item.make_sample(config, random.Random(7))
    capture = sample.capture_rgb_indices
    query = tuple(
        index for block in sample.query_rgb_blocks for index in block
    )
    assert all(
        right - left == 1
        for left, right in zip(query, query[1:])
    )
    assert set(capture).isdisjoint(query)
    assert sample.local_window_start <= query[0]
    assert query[-1] <= sample.local_window_end
    assert query[0] - sample.local_window_start == 20
    assert sample.local_window_end - query[-1] == 20
    assert len(sample.query_rgb_blocks) == 1
    assert len(sample.query_rgb_blocks[0]) == 41
    assert 2 <= len(capture) <= 24
    assert config.target_latent_frames == 41
    assert 0.0 < sample.retrieval_coverage_score <= 1.0


def test_pose_facility_retrieval_is_deterministic_for_seed() -> None:
    item = VipeRoomTourItem.__new__(VipeRoomTourItem)
    item.root = Path("/fake/scene.mp4")
    item.indices = list(range(200))
    item.pose_by_index = {}
    for index in item.indices:
        pose = np.eye(4, dtype=np.float32)
        pose[0, 3] = index / 100.0
        item.pose_by_index[index] = pose
    config = GeometryMemorySampleConfig()
    first = item.make_sample(config, random.Random(11))
    second = item.make_sample(config, random.Random(11))
    assert first.capture_rgb_indices == second.capture_rgb_indices
    assert first.query_rgb_blocks == second.query_rgb_blocks


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


def test_compact_expand_restores_real_lingbot_patch_grid_shape() -> None:
    hidden = torch.randn(2, 3, 30, 52, 32)
    compact = CompactLinear(32, 2, expand=False)
    expand = CompactLinear(32, 2, expand=True)
    compacted = compact(hidden)
    restored = expand(compacted)
    assert compacted.shape == (2, 3, 15, 26, 32)
    assert restored.shape == hidden.shape


def test_compact_attention_broadcasts_shared_rope_over_batch() -> None:
    attention = CompactSelfAttention(
        hidden_size=32,
        num_heads=4,
        norm_eps=1e-6,
        axes_dims=(2, 2, 4),
        axes_lens=(32, 32, 32),
        rope_theta=256.0,
    )
    hidden = torch.randn(2, 12, 32, requires_grad=True)
    position_ids = torch.stack(
        torch.meshgrid(
            torch.arange(3, dtype=torch.int32),
            torch.arange(2, dtype=torch.int32),
            torch.arange(2, dtype=torch.int32),
            indexing="ij",
        ),
        dim=-1,
    ).flatten(0, 2)
    output = attention(hidden, position_ids)
    assert output.shape == hidden.shape
    output.square().mean().backward()
    assert hidden.grad is not None


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


def test_direct_memory_preserves_every_history_patch_token() -> None:
    backbone = LingBotVideoTransformer3DModel(
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
    model = GIMWorldLingBotModel(
        backbone,
        GIMWorldModelConfig(image_height=64, image_width=96),
    )
    history = torch.randn(1, 4, 5, 8, 12)
    cameras = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 5, 1, 1)
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 5, 1, 1)
    memory = model.build_memory(history, cameras, intrinsics)
    expected = model.patchify_history(history)
    assert memory.shape == (1, 5 * 4 * 6, 32)
    torch.testing.assert_close(memory, expected)


def test_backbone_lora_freezes_base_and_updates_only_adapters() -> None:
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
    target_modules = (
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    summary = inject_backbone_lora(
        model,
        BackboneLoRAConfig(
            rank=4,
            alpha=4.0,
            target_modules=target_modules,
        ),
    )
    assert summary.module_count == 2 * len(target_modules)
    assert isinstance(model.blocks[0].attn.to_q, LoRALinear)
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert trainable
    assert all(".lora_a." in name or ".lora_b." in name for name in trainable)

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
    output.square().mean().backward()
    assert model.blocks[0].attn.to_q.lora_b.weight.grad is not None
    assert model.blocks[0].attn.to_q.base_layer.weight.grad is None


def test_fused_qkv_setting_does_not_bypass_lora(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    inject_backbone_lora(
        model,
        BackboneLoRAConfig(
            rank=4,
            alpha=4.0,
            target_modules=("to_q", "to_k", "to_v"),
        ),
    )
    monkeypatch.setenv("LINGBOT_FUSED_QKV_LINEAR", "1")
    target = torch.randn(1, 4, 3, 8, 12)
    text = torch.randn(1, 5, 16)
    output = model(
        target,
        torch.tensor([500.0]),
        text,
        return_dict=False,
    )[0]
    output.mean().backward()
    assert model.blocks[0].attn.to_q.lora_b.weight.grad is not None


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
