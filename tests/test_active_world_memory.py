from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from lingbot_video.active_world_memory.agent import QUERY_TYPES
from lingbot_video.active_world_memory.geometry import pairwise_camera_relevance
from lingbot_video.active_world_memory.lora import (
    LoRALinear,
    inject_lora,
    load_lora_state_dict,
    lora_state_dict,
)
from lingbot_video.active_world_memory.model import (
    ActiveWorldMemoryConfig,
    ActiveWorldMemoryModel,
)


class DummyBackbone(nn.Module):
    def __init__(self, channels: int, text_dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(in_channels=channels, out_channels=channels)
        self.condition = nn.Linear(text_dim, channels)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=False,
    ):
        value = self.condition(encoder_hidden_states.mean(dim=1))
        value = value[:, :, None, None, None].expand_as(hidden_states)
        return (value,)


def _model() -> ActiveWorldMemoryModel:
    config = ActiveWorldMemoryConfig(
        memory_dim=32,
        num_heads=4,
        coarse_grid_h=2,
        coarse_grid_w=2,
        canvas_grid_h=2,
        canvas_grid_w=2,
        condition_tokens=8,
        latent_channels=4,
        text_dim=64,
        max_retrieval_steps=4,
        min_selected_views=2,
        max_selected_views=4,
        coarse_encoder_chunk=3,
    )
    return ActiveWorldMemoryModel(DummyBackbone(4, 64), config)


def _cameras(views: int):
    c2w = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, views, 1, 1)
    c2w[0, :, 0, 3] = torch.arange(views).float() * 0.1
    intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, views, 1, 1)
    intrinsics[..., 0, 0] = 30
    intrinsics[..., 1, 1] = 30
    intrinsics[..., 0, 2] = 16
    intrinsics[..., 1, 2] = 16
    times = torch.arange(views).float().unsqueeze(0)
    return c2w, intrinsics, times


def test_agent_rollout_condition_and_dynamic_append():
    torch.manual_seed(0)
    model = _model()
    c2w, intrinsics, times = _cameras(6)
    rgb = torch.randint(0, 256, (1, 6, 3, 32, 32), dtype=torch.uint8)
    episode_ids = torch.tensor([[0, 0, 1, 1, 2, 2]])
    memory = model.build_memory_index(
        rgb, c2w, intrinsics, times, episode_ids, (32, 32)
    )
    canvas, target_camera = model.target_canvas(
        c2w[:, :3], intrinsics[:, :3], times[:, :3], (32, 32), 3
    )
    state = model.agent.initialize(canvas, memory.view_tokens.shape[1])
    region_zero = model.agent.policy_logits(
        state,
        memory,
        query_type=torch.tensor([0]),
        region=torch.tensor([0]),
    )
    region_one = model.agent.policy_logits(
        state,
        memory,
        query_type=torch.tensor([0]),
        region=torch.tensor([1]),
    )
    assert not torch.allclose(region_zero.view_logits, region_one.view_logits)
    rollout = model.agent.rollout(canvas, memory, deterministic=True)
    assert 2 <= len(rollout.selected_views) <= 4
    assert len(set(rollout.selected_views)) == len(rollout.selected_views)
    assert all(0 <= value < len(QUERY_TYPES) for value in rollout.query_types)

    condition = model.make_condition_tokens(
        rollout, memory, target_camera
    )
    assert condition.shape == (1, 3 + 8, 64)
    prompt = torch.randn(1, 5, 64)
    mask = torch.ones(1, 5, dtype=torch.long)
    noisy = torch.randn(1, 4, 3, 8, 8)
    prediction = model.generator_forward(
        noisy, torch.tensor([500.0]), prompt, mask, condition
    )
    assert prediction.shape == noisy.shape

    latent_views = torch.randn(1, 2, 4, 8, 8)
    camera_tokens = model.encode_cameras(
        c2w[:, :2], intrinsics[:, :2], (32, 32), times[:, :2]
    )
    updated = model.append_latent_memory(memory, latent_views, camera_tokens)
    assert updated.view_tokens.shape[1] == 8
    assert updated.episode_tokens.shape[1] == 4
    assert updated.view_provenance[0, -2:].eq(1).all()
    assert torch.allclose(
        updated.view_confidence[0, -2:], torch.full((2,), 0.5)
    )


def test_policy_has_no_target_feature_input():
    torch.manual_seed(7)
    model = _model().eval()
    c2w, intrinsics, times = _cameras(5)
    rgb = torch.randint(0, 256, (1, 5, 3, 32, 32), dtype=torch.uint8)
    memory = model.build_memory_index(
        rgb,
        c2w,
        intrinsics,
        times,
        torch.tensor([[0, 0, 1, 1, 2]]),
        (32, 32),
    )
    canvas, _ = model.target_canvas(
        c2w[:, :3], intrinsics[:, :3], times[:, :3], (32, 32), 3
    )
    first = model.agent.rollout(canvas, memory, deterministic=True)
    # There is deliberately no target-feature argument in the inference path.
    second = model.agent.rollout(canvas, memory, deterministic=True)
    assert first.selected_views == second.selected_views
    assert first.query_types == second.query_types


def test_camera_teacher_shape_and_range():
    c2w, _, _ = _cameras(6)
    relevance = pairwise_camera_relevance(c2w[:, :2], c2w[:, 2:])
    assert relevance.shape == (1, 2, 4)
    assert torch.all((relevance >= 0) & (relevance <= 1))


def test_native_lora_round_trip():
    torch.manual_seed(11)
    module = nn.Module()
    module.attn = nn.Module()
    module.attn.to_q = nn.Linear(8, 8, bias=False)
    value = torch.randn(2, 8)
    expected = module.attn.to_q(value).detach()
    report = inject_lora(
        module,
        rank=2,
        alpha=2.0,
        dropout=0.0,
        target_suffixes=["attn.to_q"],
    )
    assert report.modules == ("attn.to_q",)
    assert isinstance(module.attn.to_q, LoRALinear)
    assert torch.allclose(module.attn.to_q(value), expected)
    with torch.no_grad():
        module.attn.to_q.lora_up.fill_(0.25)
    state = lora_state_dict(module)

    second = nn.Module()
    second.attn = nn.Module()
    second.attn.to_q = nn.Linear(8, 8, bias=False)
    inject_lora(
        second,
        rank=2,
        alpha=2.0,
        dropout=0.0,
        target_suffixes=["attn.to_q"],
    )
    load_lora_state_dict(second, state)
    assert torch.allclose(second.attn.to_q.lora_up, module.attn.to_q.lora_up)
    assert torch.allclose(second.attn.to_q.lora_down, module.attn.to_q.lora_down)
