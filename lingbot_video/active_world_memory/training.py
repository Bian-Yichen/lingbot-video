from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .agent import (
    QUERY_TYPES,
    ActiveMemoryRollout,
    EvidenceState,
    HierarchicalMemoryIndex,
    PolicyLogits,
)
from .data import ActiveMemorySample
from .geometry import pairwise_camera_relevance
from .model import ActiveWorldMemoryModel
from .vae import encode_independent_views, encode_target_video


TrainingStage = Literal["retriever", "imitation", "policy", "joint"]


@dataclass(frozen=True)
class ActiveWorldTrainingConfig:
    stage: TrainingStage = "joint"
    timestep_shift: float = 1.0
    retrieval_loss_weight: float = 1.0
    critic_loss_weight: float = 1.0
    imitation_loss_weight: float = 1.0
    policy_loss_weight: float = 0.1
    flow_loss_weight: float = 1.0
    uncertainty_loss_weight: float = 0.1
    consistency_loss_weight: float = 0.05
    entropy_weight: float = 0.001
    condition_dropout_probability: float = 0.1
    bootstrap_views: int = 4
    imitation_steps: int = 6
    oracle_candidates: int = 8
    policy_rollouts: int = 4
    evidence_vae_chunk: int = 8
    dynamic_update_probability: float = 0.5
    predicted_dynamic_update_probability: float = 0.5

    def validate(self) -> None:
        if self.stage not in {"retriever", "imitation", "policy", "joint"}:
            raise ValueError(f"unsupported training stage {self.stage!r}")
        if min(
            self.bootstrap_views,
            self.imitation_steps,
            self.oracle_candidates,
            self.policy_rollouts,
            self.evidence_vae_chunk,
        ) < 1:
            raise ValueError("training counts must be positive")
        for name in (
            "condition_dropout_probability",
            "dynamic_update_probability",
            "predicted_dynamic_update_probability",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")


def shifted_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    if shift == 1.0:
        return sigma
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _batch_sample(
    sample: ActiveMemorySample,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "candidate_rgb": sample.candidate_rgb_uint8.unsqueeze(0),
        "candidate_c2w": sample.candidate_c2w.unsqueeze(0).to(device),
        "candidate_k": sample.candidate_intrinsics.unsqueeze(0).to(device),
        "candidate_times": sample.candidate_times.unsqueeze(0).to(device),
        "episode_ids": sample.candidate_episode_ids.unsqueeze(0).to(device),
    }


def _target_block(
    sample: ActiveMemorySample,
    block_index: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        sample.target_rgb_uint8_blocks[block_index].unsqueeze(0),
        sample.target_c2w_blocks[block_index].unsqueeze(0).to(device),
        sample.target_intrinsics_blocks[block_index].unsqueeze(0).to(device),
        sample.target_times_blocks[block_index].unsqueeze(0).to(device),
    )


def _retrieval_teacher_losses(
    model: ActiveWorldMemoryModel,
    state: EvidenceState,
    memory: HierarchicalMemoryIndex,
    target_c2w: torch.Tensor,
    candidate_c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    relevance = pairwise_camera_relevance(target_c2w, candidate_c2w).amax(dim=1)
    if relevance.shape[1] < memory.view_tokens.shape[1]:
        relevance = F.pad(
            relevance,
            (0, memory.view_tokens.shape[1] - relevance.shape[1]),
        )
    relevance = relevance * memory.view_mask.float()
    target_distribution = relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    logits = model.agent.policy_logits(state, memory)
    capture_mask = memory.view_mask & memory.view_provenance.eq(0)
    teacher_view_logits = logits.view_logits.masked_fill(
        ~capture_mask, torch.finfo(logits.view_logits.dtype).min
    )
    view_loss = -(
        target_distribution * teacher_view_logits.log_softmax(dim=-1)
    ).sum(dim=-1).mean()

    # Camera geometry is intentionally evaluated in fp32, while the policy
    # logits can be bf16/fp16 under Accelerator autocast.  scatter_add_ does
    # not perform implicit dtype conversion, so keep the teacher aggregation
    # in the relevance dtype instead of inheriting the logits dtype.
    episode_target = torch.zeros_like(
        logits.episode_logits, dtype=relevance.dtype
    )
    episode_target.scatter_add_(1, memory.view_episode_ids, relevance)
    episode_target = episode_target / episode_target.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    capture_episode_count = torch.zeros_like(
        logits.episode_logits, dtype=torch.float32
    )
    capture_episode_count.scatter_add_(
        1, memory.view_episode_ids, capture_mask.float()
    )
    teacher_episode_logits = logits.episode_logits.masked_fill(
        capture_episode_count.eq(0), torch.finfo(logits.episode_logits.dtype).min
    )
    episode_loss = -(
        episode_target * teacher_episode_logits.log_softmax(dim=-1)
    ).sum(dim=-1).mean()
    return view_loss + episode_loss, relevance


def _bootstrap_rollout(
    model: ActiveWorldMemoryModel,
    target_canvas: torch.Tensor,
    memory: HierarchicalMemoryIndex,
    relevance: torch.Tensor,
    count: int,
) -> ActiveMemoryRollout:
    state = model.agent.initialize(target_canvas, memory.view_tokens.shape[1])
    rollout = ActiveMemoryRollout(state=state)
    order = relevance[0].argsort(descending=True)
    for view in order[: min(count, int(memory.view_mask.sum().item()))].tolist():
        state = model.agent.update(state, memory, int(view))
        rollout.state = state
        rollout.selected_views.append(int(view))
    return rollout


def _imitation_rollout(
    model: ActiveWorldMemoryModel,
    target_canvas: torch.Tensor,
    target_features: torch.Tensor,
    memory: HierarchicalMemoryIndex,
    config: ActiveWorldTrainingConfig,
) -> tuple[ActiveMemoryRollout, torch.Tensor]:
    state = model.agent.initialize(target_canvas, memory.view_tokens.shape[1])
    rollout = ActiveMemoryRollout(state=state)
    losses: list[torch.Tensor] = []
    for _ in range(config.imitation_steps):
        logits = model.agent.policy_logits(state, memory)
        unselected = memory.view_mask & ~state.selected_mask
        available = int(unselected.sum().item())
        if available == 0:
            break

        # GT target features supervise the oracle only.  They never enter the
        # policy state.  The region target is where the current world belief
        # is most wrong, which makes SEARCH_PATCH a causally meaningful action.
        with torch.no_grad():
            region_error = (
                model.world_critic(state.canvas).float() - target_features.float()
            ).pow(2).mean(dim=-1)
            oracle_region = region_error.argmax(dim=-1)

            # Give every learned interrogation type a chance to propose a
            # candidate.  Their union is counterfactually evaluated by the
            # World Critic, keeping the oracle budget bounded.
            per_type = max(1, config.oracle_candidates // len(QUERY_TYPES))
            typed_proposals: list[torch.Tensor] = []
            typed_scores: list[PolicyLogits] = []
            for query_index in range(len(QUERY_TYPES)):
                query_type = torch.tensor(
                    [query_index], device=state.canvas.device, dtype=torch.long
                )
                typed = model.agent.policy_logits(
                    state,
                    memory,
                    query_type=query_type,
                    region=oracle_region,
                )
                candidate_logits = typed.view_logits.masked_fill(
                    ~unselected, float("-inf")
                )
                typed_proposals.append(
                    candidate_logits.topk(
                        min(per_type, available), dim=-1
                    ).indices.reshape(-1)
                )
                typed_scores.append(typed)
            candidates = torch.unique(torch.cat(typed_proposals), sorted=False)
            oracle_view, _ = model.agent.counterfactual_oracle(
                state,
                memory,
                candidates,
                model.world_critic,
                target_features,
            )
        stop_target = torch.tensor(
            [1.0 if oracle_view is None else 0.0], device=state.canvas.device
        )
        step_loss = F.binary_cross_entropy_with_logits(
            logits.stop_logit.float(), stop_target
        )
        if oracle_view is None:
            losses.append(step_loss)
            break
        oracle = torch.tensor([oracle_view], device=state.canvas.device)
        oracle_episode = memory.view_episode_ids[:, oracle_view]

        # Route the oracle view through the query type that currently explains
        # it best.  Stage 3 subsequently optimizes these discrete tools by
        # actual information gain rather than freezing this bootstrap routing.
        with torch.no_grad():
            query_scores = torch.stack(
                [
                    typed.view_logits[:, oracle_view]
                    + typed.episode_logits.gather(
                        1, oracle_episode.reshape(-1, 1)
                    ).squeeze(1)
                    for typed in typed_scores
                ],
                dim=-1,
            )
            oracle_query_type = query_scores.argmax(dim=-1)
        typed_logits = model.agent.policy_logits(
            state,
            memory,
            query_type=oracle_query_type,
            region=oracle_region,
        )
        step_loss = step_loss + F.cross_entropy(
            logits.query_type_logits.float(), oracle_query_type
        )
        step_loss = step_loss + F.cross_entropy(
            logits.region_logits.float(), oracle_region
        )
        step_loss = step_loss + F.cross_entropy(
            typed_logits.view_logits.float(), oracle
        )
        step_loss = step_loss + F.cross_entropy(
            typed_logits.episode_logits.float(), oracle_episode
        )
        losses.append(step_loss)
        state = model.agent.update(state, memory, oracle_view)
        rollout.state = state
        rollout.selected_views.append(oracle_view)
        rollout.query_types.append(int(oracle_query_type.item()))
        rollout.regions.append(int(oracle_region.item()))
    if not losses:
        imitation_loss = target_canvas.new_zeros(())
    else:
        imitation_loss = torch.stack(losses).mean()
    return rollout, imitation_loss


def _policy_rollouts(
    model: ActiveWorldMemoryModel,
    target_canvas: torch.Tensor,
    target_features: torch.Tensor,
    memory: HierarchicalMemoryIndex,
    config: ActiveWorldTrainingConfig,
) -> tuple[list[ActiveMemoryRollout], torch.Tensor, torch.Tensor]:
    rollouts = [
        model.agent.rollout(
            target_canvas,
            memory,
            deterministic=False,
            critic=model.world_critic,
            target_features=target_features,
        )
        for _ in range(config.policy_rollouts)
    ]
    rewards = torch.stack([rollout.total_reward for rollout in rollouts])
    advantages = (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(1e-4)
    policy_loss = -torch.stack(
        [
            advantage.detach() * rollout.total_log_prob
            for advantage, rollout in zip(advantages, rollouts, strict=True)
        ]
    ).mean()
    entropies = [entropy for rollout in rollouts for entropy in rollout.entropies]
    entropy = torch.stack(entropies).mean() if entropies else policy_loss.new_zeros(())
    return rollouts, policy_loss, entropy


def _critic_losses(
    model: ActiveWorldMemoryModel,
    rollout: ActiveMemoryRollout,
    target_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    prediction = model.world_critic(rollout.state.canvas)
    critic_loss = F.mse_loss(prediction.float(), target_features.float())
    region_error = (prediction.detach().float() - target_features.float()).pow(2).mean(dim=-1)
    normalized_error = region_error / region_error.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    uncertainty_loss = F.mse_loss(
        rollout.state.uncertainty.float(), normalized_error
    )
    return critic_loss, uncertainty_loss


def _selected_fine_evidence(
    model: ActiveWorldMemoryModel,
    vae: torch.nn.Module,
    sample: ActiveMemorySample,
    rollout: ActiveMemoryRollout,
    candidate_camera_tokens: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    chunk_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    source_count = sample.candidate_rgb_uint8.shape[0]
    source_views = [index for index in rollout.selected_views if index < source_count]
    if not source_views:
        return None, None
    indices_cpu = torch.tensor(source_views, dtype=torch.long)
    rgb = sample.candidate_rgb_uint8.index_select(0, indices_cpu).unsqueeze(0)
    latents = encode_independent_views(
        vae, rgb, device=device, dtype=dtype, chunk_size=chunk_size
    )
    indices_device = indices_cpu.to(candidate_camera_tokens.device)
    camera = candidate_camera_tokens.index_select(1, indices_device)
    return model.encode_fine_latents(latents, camera)


def _active_world_training_step_impl(
    model: ActiveWorldMemoryModel,
    vae: torch.nn.Module,
    sample: ActiveMemorySample,
    *,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    config: ActiveWorldTrainingConfig,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """One scene iteration; target RGB never enters the retrieval policy state."""

    config.validate()
    batch = _batch_sample(sample, device)
    memory = model.build_memory_index(
        batch["candidate_rgb"],
        batch["candidate_c2w"],
        batch["candidate_k"],
        batch["candidate_times"],
        batch["episode_ids"],
        sample.image_hw,
    )
    candidate_camera_tokens = model.encode_cameras(
        batch["candidate_c2w"],
        batch["candidate_k"],
        sample.image_hw,
        batch["candidate_times"],
    )

    total_losses: list[torch.Tensor] = []
    flow_losses: list[torch.Tensor] = []
    retrieval_losses: list[torch.Tensor] = []
    critic_losses: list[torch.Tensor] = []
    imitation_losses: list[torch.Tensor] = []
    policy_losses: list[torch.Tensor] = []
    selected_counts: list[torch.Tensor] = []
    reward_values: list[torch.Tensor] = []
    sigmas: list[torch.Tensor] = []

    for block_index in range(len(sample.target_rgb_uint8_blocks)):
        target_rgb, target_c2w, target_k, target_times = _target_block(
            sample, block_index, device
        )
        target_latents = encode_target_video(
            vae,
            target_rgb,
            device=device,
            dtype=compute_dtype,
        )
        target_canvas, target_camera_tokens = model.target_canvas(
            target_c2w,
            target_k,
            target_times,
            sample.image_hw,
            target_latents.shape[2],
        )
        target_features = model.target_features(target_latents)
        initial_state = model.agent.initialize(target_canvas, memory.view_tokens.shape[1])
        retrieval_loss, relevance = _retrieval_teacher_losses(
            model,
            initial_state,
            memory,
            target_c2w,
            batch["candidate_c2w"],
        )

        imitation_loss = target_latents.new_zeros(())
        policy_loss = target_latents.new_zeros(())
        entropy = target_latents.new_zeros(())
        reward = target_latents.new_zeros(())
        if config.stage == "retriever":
            rollout = _bootstrap_rollout(
                model, target_canvas, memory, relevance, config.bootstrap_views
            )
        elif config.stage == "imitation":
            rollout, imitation_loss = _imitation_rollout(
                model, target_canvas, target_features, memory, config
            )
        else:
            rollouts, policy_loss, entropy = _policy_rollouts(
                model, target_canvas, target_features, memory, config
            )
            reward_tensor = torch.stack([value.total_reward for value in rollouts])
            # Do not use GT reward to choose the generator condition.  The
            # first policy sample is on-policy; all rollouts still contribute
            # to the group-relative policy gradient.
            rollout = rollouts[0]
            reward = reward_tensor.mean().detach()
            if config.imitation_loss_weight > 0:
                _, imitation_loss = _imitation_rollout(
                    model, target_canvas, target_features, memory, config
                )

        critic_loss, uncertainty_loss = _critic_losses(
            model, rollout, target_features
        )
        fine_views, fine_patches = _selected_fine_evidence(
            model,
            vae,
            sample,
            rollout,
            candidate_camera_tokens,
            device=device,
            dtype=compute_dtype,
            chunk_size=config.evidence_vae_chunk,
        )
        condition_tokens = model.make_condition_tokens(
            rollout,
            memory,
            target_camera_tokens,
            fine_views,
            fine_patches,
        )

        noise = torch.randn_like(target_latents)
        sigma = shifted_sigma(
            torch.rand(target_latents.shape[0], device=device), config.timestep_shift
        )
        sigma_view = sigma.view(-1, 1, 1, 1, 1).to(target_latents.dtype)
        noisy = (1.0 - sigma_view) * target_latents + sigma_view * noise
        velocity_target = noise - target_latents
        drop_condition = bool(
            torch.rand((), device=device).item()
            < config.condition_dropout_probability
        )
        prediction = model.generator_forward(
            noisy,
            sigma * 1000.0,
            prompt_embeds,
            prompt_mask,
            condition_tokens,
            drop_condition=drop_condition,
        )
        flow_loss = F.mse_loss(prediction.float(), velocity_target.float())

        # Two independent evidence policies for the same target should maintain
        # one world prediction.  This is cheap and directly supports multi-
        # trajectory consistency without rendering a global point cloud.
        consistency_loss = target_latents.new_zeros(())
        if config.consistency_loss_weight > 0 and config.stage in {"policy", "joint"}:
            second = model.agent.rollout(
                target_canvas,
                memory,
                deterministic=False,
                critic=model.world_critic,
                target_features=target_features,
            )
            first_prediction = model.world_critic(rollout.state.canvas)
            second_prediction = model.world_critic(second.state.canvas)
            consistency_loss = F.mse_loss(
                first_prediction.float(), second_prediction.float()
            )

        loss = (
            config.flow_loss_weight * flow_loss
            + config.retrieval_loss_weight * retrieval_loss
            + config.critic_loss_weight * critic_loss
            + config.uncertainty_loss_weight * uncertainty_loss
            + config.imitation_loss_weight * imitation_loss
            + config.policy_loss_weight * policy_loss
            + config.consistency_loss_weight * consistency_loss
            - config.entropy_weight * entropy
        )
        total_losses.append(loss)
        flow_losses.append(flow_loss.detach())
        retrieval_losses.append(retrieval_loss.detach())
        critic_losses.append(critic_loss.detach())
        imitation_losses.append(imitation_loss.detach())
        policy_losses.append(policy_loss.detach())
        selected_counts.append(
            torch.tensor(float(len(rollout.selected_views)), device=device)
        )
        reward_values.append(reward)
        sigmas.append(sigma.mean().detach())

        has_next = block_index + 1 < len(sample.target_rgb_uint8_blocks)
        if (
            has_next
            and torch.rand((), device=device).item() < config.dynamic_update_probability
        ):
            use_prediction = (
                torch.rand((), device=device).item()
                < config.predicted_dynamic_update_probability
            )
            update = (
                noisy - sigma_view * prediction.detach()
                if use_prediction
                else target_latents.detach()
            )
            latent_indices = model.latent_pose_indices(
                target_c2w.shape[1], update.shape[2], device
            )
            update_camera = model.encode_cameras(
                target_c2w[:, latent_indices],
                target_k[:, latent_indices],
                sample.image_hw,
                target_times[:, latent_indices],
            )
            memory = model.append_latent_memory(
                memory,
                update.permute(0, 2, 1, 3, 4),
                update_camera,
                confidence=(
                    (1.0 - rollout.state.uncertainty.mean())
                    .detach()
                    .clamp(0.1, 0.9)
                    if use_prediction
                    else 1.0
                ),
            )

    return {
        "loss": torch.stack(total_losses).mean(),
        "flow_loss": torch.stack(flow_losses).mean(),
        "retrieval_loss": torch.stack(retrieval_losses).mean(),
        "critic_loss": torch.stack(critic_losses).mean(),
        "imitation_loss": torch.stack(imitation_losses).mean(),
        "policy_loss": torch.stack(policy_losses).mean(),
        "selected_views": torch.stack(selected_counts).mean(),
        "information_gain_reward": torch.stack(reward_values).mean(),
        "sigma": torch.stack(sigmas).mean(),
        "memory_views": torch.tensor(float(memory.view_tokens.shape[1]), device=device),
    }


def active_world_training_step(
    model: torch.nn.Module,
    vae: torch.nn.Module,
    sample: ActiveMemorySample,
    *,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    config: ActiveWorldTrainingConfig,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """DDP-safe public entry point."""

    return model(
        operation="training_step",
        vae=vae,
        sample=sample,
        prompt_embeds=prompt_embeds,
        prompt_mask=prompt_mask,
        config=config,
        device=device,
        compute_dtype=compute_dtype,
    )
