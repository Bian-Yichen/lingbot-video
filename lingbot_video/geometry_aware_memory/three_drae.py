from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .geometry import make_origin_direction_rays
from .wan_latent_reconstruction import WanDecoderBridge


@dataclass(frozen=True)
class ThreeDRAEConfig:
    """3DRAE memory topology adapted to frozen Wan VAE latents.

    The paper's frozen 2D representation encoder and RGB unpatchification are
    deliberately replaced by the frozen Wan encoder and decoder.  Everything
    between those endpoints follows the 3DRAE representation path: ray-aware
    view tokens, a 12-layer latent fuse neck, fixed-length scene tokens, and a
    16-layer ray-query decoder.
    """

    image_height: int = 480
    image_width: int = 832
    vae_spatial_stride: int = 8
    latent_channels: int = 16
    latent_patch_size: tuple[int, int, int] = (1, 2, 2)
    hidden_size: int = 768
    num_heads: int = 16
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    num_memory_tokens: int = 1024
    encoder_depth: int = 12
    decoder_depth: int = 16
    latent_batch_norm: bool = True
    decoder_noise_tau: float = 0.8
    view_mask_probability: float = 0.1
    view_mask_ratio_min: float = 0.6
    view_mask_ratio_max: float = 0.9
    decoder_lora_rank: int = 16
    decoder_lora_alpha: float = 16.0
    decoder_refiner_hidden_size: int = 64
    vae_latents_mean: tuple[float, ...] = ()
    vae_latents_std: tuple[float, ...] = ()

    def validate(self) -> None:
        if self.image_height < 1 or self.image_width < 1:
            raise ValueError("3DRAE image dimensions must be positive")
        if (
            self.image_height % self.vae_spatial_stride
            or self.image_width % self.vae_spatial_stride
        ):
            raise ValueError("image grid must be divisible by Wan spatial stride")
        if self.latent_patch_size[0] != 1:
            raise ValueError("independent Wan views require latent patch_t=1")
        if self.hidden_size % self.num_heads:
            raise ValueError("3DRAE hidden size must be divisible by head count")
        if min(
            self.num_memory_tokens,
            self.encoder_depth,
            self.decoder_depth,
            self.hidden_size,
            self.num_heads,
        ) < 1:
            raise ValueError("3DRAE dimensions and depths must be positive")
        if self.mlp_ratio <= 0:
            raise ValueError("3DRAE MLP ratio must be positive")
        if self.decoder_noise_tau < 0:
            raise ValueError("decoder noise tau cannot be negative")
        if not 0 <= self.view_mask_probability <= 1:
            raise ValueError("view mask probability must be in [0,1]")
        if not (
            0 <= self.view_mask_ratio_min <= self.view_mask_ratio_max < 1
        ):
            raise ValueError("view mask ratios must satisfy 0 <= min <= max < 1")
        if (
            len(self.vae_latents_mean) != self.latent_channels
            or len(self.vae_latents_std) != self.latent_channels
        ):
            raise ValueError("Wan latent mean/std must match latent channels")
        latent_h = self.image_height // self.vae_spatial_stride
        latent_w = self.image_width // self.vae_spatial_stride
        _, patch_h, patch_w = self.latent_patch_size
        if latent_h % patch_h or latent_w % patch_w:
            raise ValueError("Wan latent grid is not divisible by 3DRAE patch")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThreeDRAEViewCurriculum:
    history_start_min: int = 2
    history_start_max: int = 4
    history_end_min: int = 12
    history_end_max: int = 32
    query_start: int = 2
    query_end: int = 8
    curriculum_epochs: int = 8

    def validate(self) -> None:
        if min(
            self.history_start_min,
            self.history_start_max,
            self.history_end_min,
            self.history_end_max,
            self.query_start,
            self.query_end,
            self.curriculum_epochs,
        ) < 1:
            raise ValueError("view curriculum values must be positive")
        if self.history_start_min < 2:
            raise ValueError("history curriculum requires at least two views")
        if self.history_start_min > self.history_start_max:
            raise ValueError("history start_min cannot exceed start_max")
        if self.history_end_min > self.history_end_max:
            raise ValueError("history end_min cannot exceed end_max")
        if (
            self.history_end_min < self.history_start_min
            or self.history_end_max < self.history_start_max
        ):
            raise ValueError("history view curriculum must not decrease")
        if self.query_end < self.query_start:
            raise ValueError("query view curriculum must not decrease")
        if self.query_start < 2:
            raise ValueError("room-tour query blocks require at least two views")

    def values_for_epoch(self, epoch: int) -> tuple[int, int, int]:
        self.validate()
        if self.curriculum_epochs == 1:
            progress = 1.0
        else:
            progress = min(max(epoch, 0) / (self.curriculum_epochs - 1), 1.0)

        def interpolate(start: int, end: int) -> int:
            return int(round(start + progress * (end - start)))

        return (
            interpolate(self.history_start_min, self.history_end_min),
            interpolate(self.history_start_max, self.history_end_max),
            interpolate(self.query_start, self.query_end),
        )


class ThreeDRAESelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("attention hidden size must divide into heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, hidden_size = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch,
            length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
        )
        return self.projection(
            attended.transpose(1, 2).reshape(batch, length, hidden_size)
        )


class ThreeDRAETransformerBlock(nn.Module):
    """Pre-norm global self-attention block used by both paper modules."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        intermediate_size = int(hidden_size * mlp_ratio)
        self.attention_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attention = ThreeDRAESelfAttention(hidden_size, num_heads)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.mlp(self.mlp_norm(tokens))


class PluckerRayPatchEmbedding(nn.Module):
    """Paper 7-channel Pluecker-ray plus visibility patch embedding."""

    def __init__(
        self,
        *,
        image_height: int,
        image_width: int,
        rgb_patch_height: int,
        rgb_patch_width: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        if image_height % rgb_patch_height or image_width % rgb_patch_width:
            raise ValueError("image grid is not divisible by ray patch size")
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.grid_height = image_height // rgb_patch_height
        self.grid_width = image_width // rgb_patch_width
        self.projection = nn.Conv2d(
            7,
            hidden_size,
            kernel_size=(rgb_patch_height, rgb_patch_width),
            stride=(rgb_patch_height, rgb_patch_width),
        )

    def forward(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        visibility: torch.Tensor,
    ) -> torch.Tensor:
        if c2w.ndim != 4 or c2w.shape[-2:] != (4, 4):
            raise ValueError("ray cameras must be [B,V,4,4]")
        if intrinsics.shape != (*c2w.shape[:2], 3, 3):
            raise ValueError("ray intrinsics must be [B,V,3,3]")
        if visibility.shape != c2w.shape[:2]:
            raise ValueError("ray visibility must be [B,V]")
        batch, views = c2w.shape[:2]
        rays = make_origin_direction_rays(
            c2w,
            intrinsics,
            self.image_height,
            self.image_width,
        )
        mask = visibility.to(
            device=rays.device,
            dtype=rays.dtype,
        )[..., None, None, None]
        mask = mask.expand(batch, views, self.image_height, self.image_width, 1)
        ray_map = torch.cat((rays, mask), dim=-1).reshape(
            batch * views,
            self.image_height,
            self.image_width,
            7,
        )
        embedded = self.projection(
            ray_map.permute(0, 3, 1, 2).to(self.projection.weight.dtype)
        )
        return embedded.flatten(2).transpose(1, 2).reshape(
            batch,
            views,
            self.grid_height * self.grid_width,
            -1,
        )


class ThreeDRAELatentFuseEncoder(nn.Module):
    """Paper's 12-layer Latent Fuse Neck with fixed scene queries."""

    def __init__(self, config: ThreeDRAEConfig) -> None:
        super().__init__()
        self.config = config
        self.memory_queries = nn.Parameter(
            torch.empty(1, config.num_memory_tokens, config.hidden_size)
        )
        nn.init.trunc_normal_(self.memory_queries, std=0.02)
        self.blocks = nn.ModuleList(
            [
                ThreeDRAETransformerBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.mlp_ratio,
                    config.norm_eps,
                )
                for _ in range(config.encoder_depth)
            ]
        )
        # Appendix A.2 specifies non-affine LN followed by global-stat BN.
        self.output_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
            elementwise_affine=False,
        )
        self.output_batch_norm: nn.Module | None = None
        if config.latent_batch_norm:
            self.output_batch_norm = nn.SyncBatchNorm(
                config.hidden_size,
                eps=config.norm_eps,
                momentum=None,
                affine=False,
                track_running_stats=True,
            )
        self.gradient_checkpointing = False

    def forward(self, view_tokens: torch.Tensor) -> torch.Tensor:
        if view_tokens.ndim != 4:
            raise ValueError("3DRAE view tokens must be [B,V,N,D]")
        batch, _, _, hidden_size = view_tokens.shape
        if hidden_size != self.config.hidden_size:
            raise ValueError("3DRAE view-token dimension mismatch")
        memory_queries = self.memory_queries.expand(batch, -1, -1)
        tokens = torch.cat((memory_queries, view_tokens.flatten(1, 2)), dim=1)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        memory = self.output_norm(tokens[:, : self.config.num_memory_tokens])
        if self.output_batch_norm is not None:
            memory = self.output_batch_norm(memory.transpose(1, 2)).transpose(1, 2)
        return memory


class ThreeDRAELatentQueryDecoder(nn.Module):
    """Paper's 16-layer ray-query decoder, adapted to Wan latent patches."""

    def __init__(self, config: ThreeDRAEConfig) -> None:
        super().__init__()
        self.config = config
        _, patch_h, patch_w = config.latent_patch_size
        self.latent_height = config.image_height // config.vae_spatial_stride
        self.latent_width = config.image_width // config.vae_spatial_stride
        self.patch_height = int(patch_h)
        self.patch_width = int(patch_w)
        self.blocks = nn.ModuleList(
            [
                ThreeDRAETransformerBlock(
                    config.hidden_size,
                    config.num_heads,
                    config.mlp_ratio,
                    config.norm_eps,
                )
                for _ in range(config.decoder_depth)
            ]
        )
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.output_projection = nn.Linear(
            config.hidden_size,
            config.latent_channels * patch_h * patch_w,
        )
        self.gradient_checkpointing = False

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        grid_h = self.latent_height // self.patch_height
        grid_w = self.latent_width // self.patch_width
        return (
            patches.reshape(
                batch,
                grid_h,
                grid_w,
                self.patch_height,
                self.patch_width,
                self.config.latent_channels,
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(
                batch,
                self.config.latent_channels,
                self.latent_height,
                self.latent_width,
            )
        )

    def decode_view(
        self,
        memory: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        memory_length = memory.shape[1]
        tokens = torch.cat((memory, query_tokens), dim=1)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        queries = self.output_norm(tokens[:, memory_length:])
        return self._unpatchify(self.output_projection(queries))

    def forward(
        self,
        memory: torch.Tensor,
        target_ray_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if target_ray_tokens.ndim != 4:
            raise ValueError("target ray tokens must be [B,T,N,D]")
        outputs = [
            self.decode_view(memory, target_ray_tokens[:, view_index])
            for view_index in range(target_ray_tokens.shape[1])
        ]
        return torch.stack(outputs, dim=1)


class WanThreeDRAEModel(nn.Module):
    """Frozen Wan encoder -> 3DRAE -> frozen/adapted Wan decoder."""

    def __init__(
        self,
        wan_decoder: WanDecoderBridge,
        config: ThreeDRAEConfig,
    ) -> None:
        super().__init__()
        config.validate()
        self.reconstruction_config = config
        patch_volume = math.prod(config.latent_patch_size)
        self.input_projection = nn.Linear(
            config.latent_channels * patch_volume,
            config.hidden_size,
        )
        _, patch_h, patch_w = config.latent_patch_size
        self.ray_embedding = PluckerRayPatchEmbedding(
            image_height=config.image_height,
            image_width=config.image_width,
            rgb_patch_height=config.vae_spatial_stride * patch_h,
            rgb_patch_width=config.vae_spatial_stride * patch_w,
            hidden_size=config.hidden_size,
        )
        self.memory_encoder = ThreeDRAELatentFuseEncoder(config)
        self.latent_decoder = ThreeDRAELatentQueryDecoder(config)
        self.wan_decoder = wan_decoder
        self.register_buffer(
            "vae_latents_mean",
            torch.tensor(config.vae_latents_mean).reshape(1, 1, -1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "vae_latents_std",
            torch.tensor(config.vae_latents_std).reshape(1, 1, -1, 1, 1),
            persistent=False,
        )

    @property
    def patch_grid(self) -> tuple[int, int]:
        return self.ray_embedding.grid_height, self.ray_embedding.grid_width

    @property
    def last_generator_layer(self) -> torch.Tensor:
        return self.latent_decoder.output_projection.weight

    def set_training_stage(self, stage: int) -> None:
        self.wan_decoder.set_training_stage(stage)

    def enable_gradient_checkpointing(self) -> None:
        self.memory_encoder.gradient_checkpointing = True
        self.latent_decoder.gradient_checkpointing = True
        self.wan_decoder.gradient_checkpointing = True

    def patchify_history(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError("history latents must be [B,C,V,H,W]")
        batch, channels, views, height, width = latents.shape
        patch_t, patch_h, patch_w = self.reconstruction_config.latent_patch_size
        if patch_t != 1:
            raise ValueError("history views are independent Wan samples")
        if height % patch_h or width % patch_w:
            raise ValueError("history latent grid is not divisible by patch")
        grid_h, grid_w = height // patch_h, width // patch_w
        patches = (
            latents.reshape(
                batch,
                channels,
                views,
                grid_h,
                patch_h,
                grid_w,
                patch_w,
            )
            .permute(0, 2, 3, 5, 4, 6, 1)
            .reshape(
                batch,
                views,
                grid_h * grid_w,
                channels * patch_h * patch_w,
            )
        )
        return self.input_projection(patches)

    def sample_view_visibility(
        self,
        batch: int,
        views: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        visibility = torch.ones(batch, views, dtype=torch.bool, device=device)
        if not self.training or self.reconstruction_config.view_mask_probability <= 0:
            return visibility
        for batch_index in range(batch):
            if (
                torch.rand((), device=device).item()
                >= self.reconstruction_config.view_mask_probability
            ):
                continue
            ratio = torch.empty((), device=device).uniform_(
                self.reconstruction_config.view_mask_ratio_min,
                self.reconstruction_config.view_mask_ratio_max,
            )
            masked_count = min(int(math.floor(views * float(ratio.item()))), views - 1)
            if masked_count > 0:
                indices = torch.randperm(views, device=device)[:masked_count]
                visibility[batch_index, indices] = False
        return visibility

    def build_memory(
        self,
        history_latents: torch.Tensor,
        history_c2w: torch.Tensor,
        history_intrinsics: torch.Tensor,
        *,
        view_visibility: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        appearance = self.patchify_history(history_latents)
        batch, views = appearance.shape[:2]
        if view_visibility is None:
            view_visibility = self.sample_view_visibility(
                batch,
                views,
                device=history_latents.device,
            )
        elif view_visibility.shape != (batch, views):
            raise ValueError("view visibility must match history [B,V]")
        rays = self.ray_embedding(
            history_c2w,
            history_intrinsics,
            view_visibility,
        )
        visible_features = appearance * view_visibility.to(
            appearance.dtype
        )[..., None, None]
        return self.memory_encoder(visible_features + rays), view_visibility

    def add_decoder_noise(self, memory: torch.Tensor) -> torch.Tensor:
        tau = self.reconstruction_config.decoder_noise_tau
        if not self.training or tau <= 0:
            return memory
        sigma = torch.rand(
            memory.shape[0],
            1,
            1,
            device=memory.device,
            dtype=memory.dtype,
        ) * tau
        return memory + sigma * torch.randn_like(memory)

    def normalized_to_native(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.vae_latents_std.to(latents.dtype) + (
            self.vae_latents_mean.to(latents.dtype)
        )

    def forward(
        self,
        history_latents: torch.Tensor,
        history_c2w: torch.Tensor,
        history_intrinsics: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
        *,
        view_visibility: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        memory, visibility = self.build_memory(
            history_latents,
            history_c2w,
            history_intrinsics,
            view_visibility=view_visibility,
        )
        target_visible = torch.zeros(
            target_c2w.shape[:2],
            dtype=torch.bool,
            device=target_c2w.device,
        )
        target_rays = self.ray_embedding(
            target_c2w,
            target_intrinsics,
            target_visible,
        )
        predicted_latents = self.latent_decoder(
            self.add_decoder_noise(memory),
            target_rays,
        )
        native_latents = self.normalized_to_native(predicted_latents)
        decoded_rgb = self.wan_decoder(native_latents).add(1.0).mul(0.5)
        return predicted_latents, decoded_rgb, memory, visibility

    def experiment_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = (
            "input_projection.",
            "ray_embedding.",
            "memory_encoder.",
            "latent_decoder.",
            "wan_decoder.latent_refiner.",
        )
        return {
            name: value
            for name, value in self.state_dict().items()
            if name.startswith(prefixes)
            or (
                name.startswith("wan_decoder.decoder.")
                and (".down." in name or ".up." in name)
            )
        }

    def load_experiment_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected 3DRAE weights: {unexpected}")
        required_prefixes = (
            "input_projection.",
            "ray_embedding.",
            "memory_encoder.",
            "latent_decoder.",
            "wan_decoder.latent_refiner.",
        )
        invalid_missing = [
            name
            for name in missing
            if name.startswith(required_prefixes)
            or (
                name.startswith("wan_decoder.decoder.")
                and (".down." in name or ".up." in name)
            )
        ]
        if invalid_missing:
            raise RuntimeError(
                f"checkpoint is missing 3DRAE weights: {invalid_missing}"
            )


def three_drae_config_from_wan(
    *,
    vae_latents_mean: tuple[float, ...],
    vae_latents_std: tuple[float, ...],
    **overrides: Any,
) -> ThreeDRAEConfig:
    values: dict[str, Any] = {
        "latent_channels": len(vae_latents_mean),
        "vae_latents_mean": vae_latents_mean,
        "vae_latents_std": vae_latents_std,
    }
    values.update(overrides)
    return ThreeDRAEConfig(**values)
