from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


QUERY_TYPES = (
    "structure",
    "identity",
    "appearance",
    "occlusion",
    "context",
    "loop_closure",
    "expand_temporal",
    "verify_complement",
)


@dataclass
class HierarchicalMemoryIndex:
    view_tokens: torch.Tensor  # (B,N,D)
    patch_tokens: torch.Tensor  # (B,N,P,D), inexpensive coarse inspection tokens
    episode_tokens: torch.Tensor  # (B,E,D)
    episode_mask: torch.Tensor  # (B,E)
    view_episode_ids: torch.Tensor  # (B,N)
    view_mask: torch.Tensor  # (B,N)
    view_confidence: torch.Tensor  # (B,N), capture=1, generated is calibrated
    view_provenance: torch.Tensor  # (B,N), 0=capture, 1=generated


@dataclass
class EvidenceState:
    canvas: torch.Tensor  # (B,R,D)
    uncertainty: torch.Tensor  # (B,R)
    selected_mask: torch.Tensor  # (B,N)
    steps: int = 0


@dataclass
class PolicyLogits:
    stop_logit: torch.Tensor
    query_type_logits: torch.Tensor
    region_logits: torch.Tensor
    episode_logits: torch.Tensor
    view_logits: torch.Tensor


@dataclass
class AgentDecision:
    stop: bool
    query_type: int | None = None
    region: int | None = None
    episode: int | None = None
    view: int | None = None
    log_prob: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    stop_probability: float = 0.0


@dataclass
class ActiveMemoryRollout:
    state: EvidenceState
    selected_views: list[int] = field(default_factory=list)
    query_types: list[int] = field(default_factory=list)
    regions: list[int] = field(default_factory=list)
    log_probs: list[torch.Tensor] = field(default_factory=list)
    entropies: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)
    critic_losses: list[torch.Tensor] = field(default_factory=list)
    stop_probabilities: list[float] = field(default_factory=list)

    @property
    def total_reward(self) -> torch.Tensor:
        if not self.rewards:
            return self.state.canvas.new_zeros(())
        return torch.stack(self.rewards).sum()

    @property
    def total_log_prob(self) -> torch.Tensor:
        if not self.log_probs:
            return self.state.canvas.new_zeros(())
        return torch.stack(self.log_probs).sum()


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, torch.finfo(logits.dtype).min)


def _categorical(
    logits: torch.Tensor,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distribution = torch.distributions.Categorical(logits=logits.float())
    value = logits.argmax(dim=-1) if deterministic else distribution.sample()
    return value, distribution.log_prob(value), distribution.entropy()


class EvidenceCanvasUpdater(nn.Module):
    """Inspect one retrieved view and update per-target-region belief."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.canvas_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(
            dim, num_heads, batch_first=True
        )
        self.gate = nn.Sequential(nn.Linear(3 * dim, dim), nn.Sigmoid())
        self.candidate = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.SiLU(), nn.Linear(2 * dim, dim)
        )
        self.output_norm = nn.LayerNorm(dim)
        self.uncertainty_head = nn.Linear(dim, 1)

    def forward(
        self,
        canvas: torch.Tensor,
        evidence_patches: torch.Tensor,
        evidence_view: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attended, weights = self.cross_attention(
            self.canvas_norm(canvas),
            self.evidence_norm(evidence_patches),
            evidence_patches,
            need_weights=True,
            average_attn_weights=True,
        )
        view = evidence_view.unsqueeze(1).expand_as(canvas)
        joint = torch.cat((canvas, attended, view), dim=-1)
        gate = self.gate(joint)
        proposal = self.candidate(joint)
        updated = self.output_norm(canvas + gate * proposal)
        uncertainty = torch.sigmoid(self.uncertainty_head(updated).squeeze(-1))
        return updated, uncertainty, weights


class ActiveWorldMemoryAgent(nn.Module):
    """A hierarchical visual policy with explicit retrieval and STOP actions."""

    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        max_steps: int = 6,
        min_selected_views: int = 2,
        max_selected_views: int = 24,
        retrieval_cost: float = 0.01,
    ) -> None:
        super().__init__()
        if min_selected_views < 0 or max_selected_views < min_selected_views:
            raise ValueError("invalid selected-view bounds")
        self.dim = dim
        self.max_steps = max_steps
        self.min_selected_views = min_selected_views
        self.max_selected_views = max_selected_views
        self.retrieval_cost = retrieval_cost
        self.query_type_embeddings = nn.Parameter(torch.randn(len(QUERY_TYPES), dim) * 0.02)
        self.summary_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.summary_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.region_head = nn.Linear(dim, 1)
        self.stop_head = nn.Sequential(
            nn.Linear(dim + 2, dim), nn.SiLU(), nn.Linear(dim, 1)
        )
        self.episode_key = nn.Linear(dim, dim, bias=False)
        self.view_key = nn.Linear(dim, dim, bias=False)
        self.patch_key = nn.Linear(dim, dim, bias=False)
        self.updater = EvidenceCanvasUpdater(dim, num_heads)

    def initialize(
        self,
        target_canvas: torch.Tensor,
        num_memory_views: int,
    ) -> EvidenceState:
        batch = target_canvas.shape[0]
        return EvidenceState(
            canvas=target_canvas,
            uncertainty=torch.ones(
                batch,
                target_canvas.shape[1],
                device=target_canvas.device,
                dtype=torch.float32,
            ),
            selected_mask=torch.zeros(
                batch,
                num_memory_views,
                device=target_canvas.device,
                dtype=torch.bool,
            ),
        )

    def summarize(self, state: EvidenceState) -> torch.Tensor:
        query = self.summary_query.view(1, 1, -1).expand(state.canvas.shape[0], 1, -1)
        # Uncertain regions receive a positive additive attention bias.
        bias = torch.log(state.uncertainty.clamp_min(1e-4)).unsqueeze(1)
        summary, _ = self.summary_attention(
            query,
            state.canvas,
            state.canvas,
            attn_mask=bias.repeat_interleave(self.summary_attention.num_heads, dim=0),
            need_weights=False,
        )
        return summary[:, 0]

    def policy_logits(
        self,
        state: EvidenceState,
        memory: HierarchicalMemoryIndex,
        *,
        query_type: torch.Tensor | None = None,
        region: torch.Tensor | None = None,
    ) -> PolicyLogits:
        summary = self.summarize(state)
        query_type_logits = summary @ self.query_type_embeddings.t() / math.sqrt(self.dim)
        if query_type is None:
            probabilities = query_type_logits.softmax(dim=-1)
            query_type_token = probabilities @ self.query_type_embeddings
        else:
            query_type_token = self.query_type_embeddings[query_type]
        if region is None:
            region_token = torch.zeros_like(summary)
        else:
            region_token = state.canvas.gather(
                1,
                region.reshape(-1, 1, 1).expand(-1, 1, self.dim),
            )[:, 0]
        query = F.normalize(
            self.query_projection(summary + query_type_token + region_token), dim=-1
        )

        episode_keys = F.normalize(self.episode_key(memory.episode_tokens), dim=-1)
        episode_logits = torch.einsum("bd,bed->be", query, episode_keys)
        if query_type is not None:
            # EXPAND_TEMPORAL explicitly searches neighboring episodes of
            # already accepted evidence instead of repeating a global lookup.
            for batch_index in range(state.canvas.shape[0]):
                if int(query_type[batch_index].item()) == 6 and bool(
                    state.selected_mask[batch_index].any().item()
                ):
                    selected_episodes = memory.view_episode_ids[batch_index][
                        state.selected_mask[batch_index]
                    ]
                    episode_axis = torch.arange(
                        episode_logits.shape[1], device=episode_logits.device
                    )
                    distance = (
                        episode_axis[:, None] - selected_episodes[None, :]
                    ).abs().amin(dim=1)
                    episode_logits[batch_index] -= 0.5 * distance
        episode_logits = _masked_logits(episode_logits, memory.episode_mask)
        view_keys = F.normalize(self.view_key(memory.view_tokens), dim=-1)
        view_logits = torch.einsum("bd,bnd->bn", query, view_keys)
        if region is not None:
            # SEARCH_PATCH is a coarse-to-fine action: the selected target
            # region queries every candidate's patch bank, then the best patch
            # score contributes to that view's retrieval score.
            patch_keys = F.normalize(self.patch_key(memory.patch_tokens), dim=-1)
            patch_logits = torch.einsum("bd,bnpd->bnp", query, patch_keys)
            view_logits = view_logits + patch_logits.amax(dim=-1)
        view_logits = view_logits + torch.log(
            memory.view_confidence.float().clamp_min(1e-4)
        ).to(view_logits.dtype)
        if query_type is not None:
            # VERIFY_COMPLEMENT prefers a different observation over another
            # near-duplicate of evidence already on the canvas.
            for batch_index in range(state.canvas.shape[0]):
                if int(query_type[batch_index].item()) == 7 and bool(
                    state.selected_mask[batch_index].any().item()
                ):
                    selected = view_keys[batch_index][state.selected_mask[batch_index]]
                    redundancy = view_keys[batch_index] @ selected.t()
                    view_logits[batch_index] -= redundancy.amax(dim=-1)
        view_logits = _masked_logits(
            view_logits, memory.view_mask & ~state.selected_mask
        )

        region_logits = self.region_head(state.canvas).squeeze(-1)
        region_logits = region_logits + torch.log(state.uncertainty.clamp_min(1e-4))
        selected_fraction = state.selected_mask.float().sum(dim=-1, keepdim=True) / float(
            max(1, self.max_selected_views)
        )
        selected_fraction = selected_fraction.clamp(0.0, 1.0)
        mean_uncertainty = state.uncertainty.mean(dim=-1, keepdim=True)
        stop_input = torch.cat((summary, mean_uncertainty, selected_fraction), dim=-1)
        stop_logit = self.stop_head(stop_input).squeeze(-1)
        return PolicyLogits(
            stop_logit=stop_logit,
            query_type_logits=query_type_logits,
            region_logits=region_logits,
            episode_logits=episode_logits,
            view_logits=view_logits,
        )

    def decide(
        self,
        state: EvidenceState,
        memory: HierarchicalMemoryIndex,
        *,
        deterministic: bool,
        stop_threshold: float = 0.5,
    ) -> AgentDecision:
        if state.canvas.shape[0] != 1:
            raise NotImplementedError(
                "variable-length agent rollouts currently require one scene per rank"
            )
        selected_count = int(state.selected_mask.sum().item())
        base = self.policy_logits(state, memory)
        stop_probability_tensor = torch.sigmoid(base.stop_logit)
        stop_probability = float(stop_probability_tensor.item())
        can_stop = selected_count >= self.min_selected_views
        must_stop = (
            selected_count >= self.max_selected_views
            or selected_count >= int(memory.view_mask.sum().item())
            or state.steps >= self.max_steps
        )
        wants_stop = (
            stop_probability >= stop_threshold
            if deterministic
            else bool(torch.bernoulli(stop_probability_tensor).item())
        )
        stop_distribution = torch.distributions.Bernoulli(logits=base.stop_logit.float())
        if must_stop or (can_stop and wants_stop):
            stop_value = torch.ones_like(base.stop_logit)
            return AgentDecision(
                stop=True,
                log_prob=stop_distribution.log_prob(stop_value).mean(),
                entropy=stop_distribution.entropy().mean(),
                stop_probability=stop_probability,
            )

        query_type, query_log_prob, query_entropy = _categorical(
            base.query_type_logits, deterministic=deterministic
        )
        region, region_log_prob, region_entropy = _categorical(
            base.region_logits, deterministic=deterministic
        )
        typed = self.policy_logits(
            state,
            memory,
            query_type=query_type,
            region=region,
        )
        episode, episode_log_prob, episode_entropy = _categorical(
            typed.episode_logits, deterministic=deterministic
        )
        episode_view_mask = memory.view_episode_ids.eq(episode.unsqueeze(-1))
        valid_view_mask = episode_view_mask & memory.view_mask & ~state.selected_mask
        # If an episode was exhausted due to a previous selection, fall back to
        # the global view policy rather than producing an invalid categorical.
        if not bool(valid_view_mask.any().item()):
            valid_view_mask = memory.view_mask & ~state.selected_mask
        view_logits = _masked_logits(typed.view_logits, valid_view_mask)
        view, view_log_prob, view_entropy = _categorical(
            view_logits, deterministic=deterministic
        )
        continue_value = torch.zeros_like(base.stop_logit)
        log_prob = (
            stop_distribution.log_prob(continue_value).mean()
            + query_log_prob.mean()
            + region_log_prob.mean()
            + episode_log_prob.mean()
            + view_log_prob.mean()
        )
        entropy = (
            stop_distribution.entropy().mean()
            + query_entropy.mean()
            + region_entropy.mean()
            + episode_entropy.mean()
            + view_entropy.mean()
        )
        return AgentDecision(
            stop=False,
            query_type=int(query_type.item()),
            region=int(region.item()),
            episode=int(episode.item()),
            view=int(view.item()),
            log_prob=log_prob,
            entropy=entropy,
            stop_probability=stop_probability,
        )

    def update(
        self,
        state: EvidenceState,
        memory: HierarchicalMemoryIndex,
        view_index: int,
    ) -> EvidenceState:
        patches = memory.patch_tokens[:, view_index]
        view = memory.view_tokens[:, view_index]
        canvas, uncertainty, _ = self.updater(state.canvas, patches, view)
        confidence = memory.view_confidence[:, view_index].view(-1, 1, 1)
        canvas = state.canvas + confidence.to(canvas.dtype) * (canvas - state.canvas)
        uncertainty = state.uncertainty + confidence.squeeze(-1).float() * (
            uncertainty - state.uncertainty
        )
        selected_mask = state.selected_mask.clone()
        selected_mask[:, view_index] = True
        return EvidenceState(
            canvas=canvas,
            uncertainty=uncertainty,
            selected_mask=selected_mask,
            steps=state.steps + 1,
        )

    def rollout(
        self,
        target_canvas: torch.Tensor,
        memory: HierarchicalMemoryIndex,
        *,
        deterministic: bool,
        critic: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_features: torch.Tensor | None = None,
        stop_threshold: float = 0.5,
    ) -> ActiveMemoryRollout:
        state = self.initialize(target_canvas, memory.view_tokens.shape[1])
        rollout = ActiveMemoryRollout(state=state)
        previous_critic_loss: torch.Tensor | None = None
        if critic is not None and target_features is not None:
            previous_critic_loss = F.mse_loss(
                critic(state.canvas).float(), target_features.float()
            )
            rollout.critic_losses.append(previous_critic_loss)

        while True:
            decision = self.decide(
                state,
                memory,
                deterministic=deterministic,
                stop_threshold=stop_threshold,
            )
            if decision.log_prob is not None:
                rollout.log_probs.append(decision.log_prob)
            if decision.entropy is not None:
                rollout.entropies.append(decision.entropy)
            rollout.stop_probabilities.append(decision.stop_probability)
            if decision.stop:
                break
            assert decision.view is not None
            state = self.update(state, memory, decision.view)
            rollout.state = state
            rollout.selected_views.append(decision.view)
            rollout.query_types.append(int(decision.query_type))
            rollout.regions.append(int(decision.region))

            if critic is not None and target_features is not None:
                critic_loss = F.mse_loss(
                    critic(state.canvas).float(), target_features.float()
                )
                assert previous_critic_loss is not None
                rollout.rewards.append(
                    previous_critic_loss.detach()
                    - critic_loss.detach()
                    - self.retrieval_cost
                )
                rollout.critic_losses.append(critic_loss)
                previous_critic_loss = critic_loss
        return rollout

    def counterfactual_oracle(
        self,
        state: EvidenceState,
        memory: HierarchicalMemoryIndex,
        candidate_views: torch.Tensor,
        critic: Callable[[torch.Tensor], torch.Tensor],
        target_features: torch.Tensor,
    ) -> tuple[int | None, torch.Tensor]:
        """Choose the view with maximal one-step target-feature information gain."""

        if state.canvas.shape[0] != 1:
            raise NotImplementedError("oracle evaluation expects one scene per rank")
        current = F.mse_loss(critic(state.canvas).float(), target_features.float())
        gains: list[torch.Tensor] = []
        indices: list[int] = []
        for candidate in candidate_views.reshape(-1).tolist():
            candidate = int(candidate)
            if bool(state.selected_mask[0, candidate].item()):
                continue
            updated = self.update(state, memory, candidate)
            loss = F.mse_loss(critic(updated.canvas).float(), target_features.float())
            gains.append(current.detach() - loss.detach() - self.retrieval_cost)
            indices.append(candidate)
        if not gains:
            return None, current.new_zeros(0)
        gain_tensor = torch.stack(gains)
        best = int(gain_tensor.argmax().item())
        if (
            int(state.selected_mask.sum().item()) >= self.min_selected_views
            and float(gain_tensor[best].item()) <= 0
        ):
            return None, gain_tensor
        return indices[best], gain_tensor
