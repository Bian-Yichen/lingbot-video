from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .geometry import camera_vector, make_origin_direction_rays
from .memory_encoder import GIMImplicitMemoryEncoder, GIMMemoryEncoderConfig
from .memory_reconstruction import LatentQueryDecoderBlock, LingBotPatchConfig


@dataclass(frozen=True)
class WanLatentReconstructionConfig:
    image_height: int = 480
    image_width: int = 832
    vae_spatial_stride: int = 8
    latent_channels: int = 16
    latent_patch_size: tuple[int, int, int] = (1, 2, 2)
    encoder_hidden_size: int = 1280
    encoder_num_heads: int = 10
    encoder_norm_eps: float = 1e-6
    encoder_axes_dims: tuple[int, int, int] = (32, 48, 48)
    encoder_axes_lens: tuple[int, int, int] = (8192, 1024, 1024)
    encoder_rope_theta: float = 256.0
    memory_latent_frames: int = 1
    memory_depth: int = 2
    compact_stride: int = 2
    memory_intermediate_ratio: float = 4.0
    renderer_hidden_size: int = 768
    renderer_depth: int = 8
    renderer_num_heads: int = 16
    renderer_mlp_ratio: float = 4.0
    renderer_norm_eps: float = 1e-6
    decoder_lora_rank: int = 16
    decoder_lora_alpha: float = 16.0
    decoder_refiner_hidden_size: int = 64
    train_patch_embedder: bool = False
    vae_latents_mean: tuple[float, ...] = ()
    vae_latents_std: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoryViewCurriculum:
    start_min: int = 2
    start_max: int = 4
    end_min: int = 12
    end_max: int = 32
    curriculum_epochs: int = 8

    def validate(self) -> None:
        if min(self.start_min, self.start_max, self.end_min, self.end_max) < 2:
            raise ValueError("history curriculum requires at least two views")
        if self.start_min > self.start_max:
            raise ValueError("history start_min cannot exceed start_max")
        if self.end_min > self.end_max:
            raise ValueError("history end_min cannot exceed end_max")
        if self.end_min < self.start_min or self.end_max < self.start_max:
            raise ValueError("history curriculum must not decrease view counts")
        if self.curriculum_epochs < 1:
            raise ValueError("history curriculum epochs must be positive")

    def views_for_epoch(self, epoch: int) -> tuple[int, int]:
        self.validate()
        if self.curriculum_epochs == 1:
            progress = 1.0
        else:
            progress = min(max(epoch, 0) / (self.curriculum_epochs - 1), 1.0)
        minimum = round(
            self.start_min + progress * (self.end_min - self.start_min)
        )
        maximum = round(
            self.start_max + progress * (self.end_max - self.start_max)
        )
        return int(minimum), int(maximum)


class PluckerWanLatentRenderer(nn.Module):
    """Render a camera-aligned Wan latent grid from view-decoupled memory."""

    def __init__(
        self,
        *,
        memory_hidden_size: int,
        latent_channels: int,
        image_height: int,
        image_width: int,
        vae_spatial_stride: int,
        latent_patch_height: int,
        latent_patch_width: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.latent_channels = int(latent_channels)
        self.latent_height = image_height // vae_spatial_stride
        self.latent_width = image_width // vae_spatial_stride
        self.latent_patch_height = int(latent_patch_height)
        self.latent_patch_width = int(latent_patch_width)
        self.rgb_patch_height = vae_spatial_stride * latent_patch_height
        self.rgb_patch_width = vae_spatial_stride * latent_patch_width
        if (
            self.latent_height % latent_patch_height
            or self.latent_width % latent_patch_width
        ):
            raise ValueError("Wan latent grid is not divisible by renderer patch")
        self.gradient_checkpointing = False
        self.memory_projection = nn.Linear(memory_hidden_size, hidden_size)
        self.ray_patch_embedding = nn.Conv2d(
            7,
            hidden_size,
            kernel_size=(self.rgb_patch_height, self.rgb_patch_width),
            stride=(self.rgb_patch_height, self.rgb_patch_width),
        )
        self.blocks = nn.ModuleList(
            [
                LatentQueryDecoderBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    norm_eps,
                )
                for _ in range(depth)
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.output_projection = nn.Linear(
            hidden_size,
            latent_channels * latent_patch_height * latent_patch_width,
        )

    def _ray_queries(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        rays = make_origin_direction_rays(
            c2w,
            intrinsics,
            self.image_height,
            self.image_width,
        )
        visibility = torch.zeros_like(rays[..., :1])
        ray_map = torch.cat((rays, visibility), dim=-1).permute(0, 3, 1, 2)
        queries = self.ray_patch_embedding(
            ray_map.to(dtype=self.ray_patch_embedding.weight.dtype)
        )
        return queries.flatten(2).transpose(1, 2)

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch = patches.shape[0]
        grid_h = self.latent_height // self.latent_patch_height
        grid_w = self.latent_width // self.latent_patch_width
        return (
            patches.reshape(
                batch,
                grid_h,
                grid_w,
                self.latent_patch_height,
                self.latent_patch_width,
                self.latent_channels,
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(
                batch,
                self.latent_channels,
                self.latent_height,
                self.latent_width,
            )
        )

    def decode_view(
        self,
        memory: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        memory_tokens = self.memory_projection(memory)
        query_tokens = self._ray_queries(c2w, intrinsics)
        memory_length = memory_tokens.shape[1]
        tokens = torch.cat((memory_tokens, query_tokens), dim=1)
        for block in self.blocks:
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        query_output = self.output_norm(tokens[:, memory_length:])
        return self._unpatchify(self.output_projection(query_output))

    def forward(
        self,
        memory: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        outputs = [
            self.decode_view(
                memory,
                target_c2w[:, view_index],
                target_intrinsics[:, view_index],
            )
            for view_index in range(target_c2w.shape[1])
        ]
        return torch.stack(outputs, dim=1)


class LoRAConv2d1x1(nn.Module):
    """LoRA for Wan mid-block 1x1 attention projections."""

    def __init__(
        self,
        base: nn.Conv2d,
        *,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if base.kernel_size != (1, 1) or base.stride != (1, 1):
            raise ValueError("Wan attention LoRA requires a stride-1 1x1 conv")
        if rank < 1:
            raise ValueError("decoder LoRA rank must be positive")
        self.base = base.requires_grad_(False)
        self.down = nn.Conv2d(base.in_channels, rank, 1, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        self.scale = float(alpha) / float(rank)
        self.stage_scale = 0.0
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.down(hidden_states))
        return self.base(hidden_states) + (
            self.stage_scale * self.scale * residual
        )


class WanLatentRefiner(nn.Module):
    """Zero-initialized stage-two adapter in native Wan latent space."""

    def __init__(self, channels: int, hidden_size: int) -> None:
        super().__init__()
        self.stage_scale = 0.0
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_size, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_size, channels, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return latents + self.stage_scale * self.net(latents)


def _inject_wan_attention_lora(
    decoder: nn.Module,
    *,
    rank: int,
    alpha: float,
) -> tuple[str, ...]:
    replacements: list[tuple[str, nn.Module, str, nn.Conv2d]] = []
    for module_name, module in decoder.named_modules():
        if "attentions" not in module_name:
            continue
        for child_name, child in module.named_children():
            if child_name in {"to_qkv", "proj"} and isinstance(
                child,
                nn.Conv2d,
            ):
                full_name = f"{module_name}.{child_name}"
                replacements.append((full_name, module, child_name, child))
    if not replacements:
        raise RuntimeError("Wan decoder has no compatible attention projections")
    for _, parent, child_name, child in replacements:
        setattr(
            parent,
            child_name,
            LoRAConv2d1x1(child, rank=rank, alpha=alpha),
        )
    return tuple(name for name, _, _, _ in replacements)


class WanDecoderBridge(nn.Module):
    """Pretrained Wan inverse mapping with gated stage-two adapters."""

    def __init__(
        self,
        post_quant_conv: nn.Module,
        decoder: nn.Module,
        *,
        latent_channels: int,
        refiner_hidden_size: int,
        lora_rank: int,
        lora_alpha: float,
    ) -> None:
        super().__init__()
        self.post_quant_conv = post_quant_conv.requires_grad_(False)
        self.decoder = decoder.requires_grad_(False)
        self.lora_module_names = _inject_wan_attention_lora(
            self.decoder,
            rank=lora_rank,
            alpha=lora_alpha,
        )
        self.latent_refiner = WanLatentRefiner(
            latent_channels,
            refiner_hidden_size,
        )
        self.gradient_checkpointing = False
        self.training_stage = 1
        self.set_training_stage(1)

    def adapter_parameters(self) -> list[nn.Parameter]:
        parameters = list(self.latent_refiner.parameters())
        for module in self.modules():
            if isinstance(module, LoRAConv2d1x1):
                parameters.extend(module.down.parameters())
                parameters.extend(module.up.parameters())
        return parameters

    @property
    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.adapter_parameters())

    def set_training_stage(self, stage: int) -> None:
        if stage not in {1, 2}:
            raise ValueError("training stage must be 1 or 2")
        scale = 0.0 if stage == 1 else 1.0
        self.training_stage = int(stage)
        self.latent_refiner.stage_scale = scale
        for module in self.modules():
            if isinstance(module, LoRAConv2d1x1):
                module.stage_scale = scale

    def forward(self, native_latents: torch.Tensor) -> torch.Tensor:
        # Each novel view is an independent one-frame Wan sample. Never put
        # arbitrary views on Wan's causal temporal axis.
        if native_latents.ndim != 5:
            raise ValueError("native_latents must be [B,V,C,H,W]")
        batch, views, channels, height, width = native_latents.shape
        if channels != self.latent_refiner.net[0].in_channels:
            raise ValueError("native latent channel count does not match Wan")
        flattened = native_latents.reshape(
            batch * views,
            channels,
            height,
            width,
        )
        flattened = self.latent_refiner(flattened).unsqueeze(2)
        decoded_input = self.post_quant_conv(flattened)

        def decode_one_frame(value: torch.Tensor) -> torch.Tensor:
            # With no preceding temporal frame, `feat_cache=None` is exactly
            # equivalent to Diffusers' all-None cache. Avoiding mutable cache
            # state makes the frozen/adapted decoder safe to checkpoint.
            return self.decoder(value, first_chunk=True)

        if (
            self.gradient_checkpointing
            and self.training
            and torch.is_grad_enabled()
        ):
            decoded = checkpoint(
                decode_one_frame,
                decoded_input,
                use_reentrant=False,
            )
        else:
            decoded = decode_one_frame(decoded_input)
        decoded = decoded.clamp(-1.0, 1.0)
        if decoded.shape[2] != 1:
            raise RuntimeError(
                "one target latent must decode to one RGB frame, got "
                f"{decoded.shape[2]}"
            )
        return decoded.squeeze(2).reshape(
            batch,
            views,
            decoded.shape[1],
            decoded.shape[3],
            decoded.shape[4],
        )


class GIMWanLatentReconstructionModel(nn.Module):
    def __init__(
        self,
        patch_embedder: nn.Linear,
        wan_decoder: WanDecoderBridge,
        config: WanLatentReconstructionConfig,
    ) -> None:
        super().__init__()
        self.reconstruction_config = config
        patch_t, patch_h, patch_w = config.latent_patch_size
        if patch_t != 1:
            raise ValueError("independent-frame Wan latent patch_t must be one")
        if (
            len(config.vae_latents_mean) != config.latent_channels
            or len(config.vae_latents_std) != config.latent_channels
        ):
            raise ValueError("Wan latent mean/std must match latent channels")
        latent_h = config.image_height // config.vae_spatial_stride
        latent_w = config.image_width // config.vae_spatial_stride
        expected_in = config.latent_channels * math.prod(config.latent_patch_size)
        if (
            patch_embedder.in_features != expected_in
            or patch_embedder.out_features != config.encoder_hidden_size
        ):
            raise ValueError("patch embedder does not match Wan latent config")
        self.patch_embedder = patch_embedder
        self.patch_embedder.requires_grad_(config.train_patch_embedder)
        grid_h, grid_w = latent_h // patch_h, latent_w // patch_w
        memory_config = GIMMemoryEncoderConfig(
            hidden_size=config.encoder_hidden_size,
            num_heads=config.encoder_num_heads,
            intermediate_size=int(
                config.encoder_hidden_size * config.memory_intermediate_ratio
            ),
            memory_latent_frames=config.memory_latent_frames,
            patch_height=grid_h,
            patch_width=grid_w,
            compact_stride=config.compact_stride,
            depth=config.memory_depth,
            camera_input_dim=16,
            norm_eps=config.encoder_norm_eps,
            axes_dims=config.encoder_axes_dims,
            axes_lens=config.encoder_axes_lens,
            rope_theta=config.encoder_rope_theta,
        )
        self.memory_encoder = GIMImplicitMemoryEncoder(memory_config)
        self.latent_renderer = PluckerWanLatentRenderer(
            memory_hidden_size=config.encoder_hidden_size,
            latent_channels=config.latent_channels,
            image_height=config.image_height,
            image_width=config.image_width,
            vae_spatial_stride=config.vae_spatial_stride,
            latent_patch_height=patch_h,
            latent_patch_width=patch_w,
            hidden_size=config.renderer_hidden_size,
            depth=config.renderer_depth,
            num_heads=config.renderer_num_heads,
            mlp_ratio=config.renderer_mlp_ratio,
            norm_eps=config.renderer_norm_eps,
        )
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
        config = self.memory_encoder.config
        return config.patch_height, config.patch_width

    def set_training_stage(self, stage: int) -> None:
        self.wan_decoder.set_training_stage(stage)

    def patchify_history(self, latents: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = latents.shape
        patch_t, patch_h, patch_w = self.reconstruction_config.latent_patch_size
        grid_t = frames // patch_t
        grid_h = height // patch_h
        grid_w = width // patch_w
        patches = (
            latents.reshape(
                batch,
                channels,
                grid_t,
                patch_t,
                grid_h,
                patch_h,
                grid_w,
                patch_w,
            )
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(
                batch,
                grid_t,
                grid_h * grid_w,
                patch_t * patch_h * patch_w * channels,
            )
        )
        return self.patch_embedder(patches)

    def build_memory(
        self,
        history_latents: torch.Tensor,
        history_c2w: torch.Tensor,
        history_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        history_tokens = self.patchify_history(history_latents)
        cameras = camera_vector(
            history_c2w,
            history_intrinsics,
            (
                self.reconstruction_config.image_height,
                self.reconstruction_config.image_width,
            ),
        )
        return self.memory_encoder(history_tokens, cameras)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        memory = self.build_memory(
            history_latents,
            history_c2w,
            history_intrinsics,
        )
        predicted_latents = self.latent_renderer(
            memory,
            target_c2w,
            target_intrinsics,
        )
        native_latents = self.normalized_to_native(predicted_latents)
        decoded_rgb = self.wan_decoder(native_latents).add(1.0).mul(0.5)
        return predicted_latents, decoded_rgb, memory

    def experiment_state_dict(self) -> dict[str, torch.Tensor]:
        prefixes = (
            "patch_embedder.",
            "memory_encoder.",
            "latent_renderer.",
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
            raise RuntimeError(f"unexpected experiment weights: {unexpected}")
        required_prefixes = (
            "patch_embedder.",
            "memory_encoder.",
            "latent_renderer.",
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
                f"checkpoint is missing experiment weights: {invalid_missing}"
            )


def wan_reconstruction_config_from_lingbot(
    patch: LingBotPatchConfig,
    *,
    vae_latents_mean: tuple[float, ...],
    vae_latents_std: tuple[float, ...],
    **overrides: Any,
) -> WanLatentReconstructionConfig:
    values: dict[str, Any] = {
        "latent_channels": patch.latent_channels,
        "latent_patch_size": patch.patch_size,
        "encoder_hidden_size": patch.hidden_size,
        "encoder_num_heads": patch.num_attention_heads,
        "encoder_norm_eps": patch.norm_eps,
        "encoder_axes_dims": patch.axes_dims,
        "encoder_axes_lens": patch.axes_lens,
        "encoder_rope_theta": patch.rope_theta,
        "vae_latents_mean": vae_latents_mean,
        "vae_latents_std": vae_latents_std,
    }
    values.update(overrides)
    return WanLatentReconstructionConfig(**values)
