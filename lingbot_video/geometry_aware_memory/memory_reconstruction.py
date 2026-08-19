from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.checkpoint import checkpoint

from .geometry import camera_vector, make_origin_direction_rays
from .memory_encoder import GIMImplicitMemoryEncoder, GIMMemoryEncoderConfig


@dataclass(frozen=True)
class LingBotPatchConfig:
    latent_channels: int
    patch_size: tuple[int, int, int]
    hidden_size: int
    num_attention_heads: int
    norm_eps: float
    axes_dims: tuple[int, int, int]
    axes_lens: tuple[int, int, int]
    rope_theta: float
    patch_embed_bias: bool


@dataclass(frozen=True)
class MemoryReconstructionConfig:
    """Standalone GIM encoder plus a 3DRAE-style RGB query decoder."""

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
    decoder_hidden_size: int = 768
    decoder_depth: int = 16
    decoder_num_heads: int = 16
    decoder_mlp_ratio: float = 4.0
    decoder_norm_eps: float = 1e-6
    train_patch_embedder: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _transformer_root(model_dir: str | Path) -> Path:
    root = Path(model_dir).expanduser()
    nested = root / "transformer"
    return nested if (nested / "config.json").is_file() else root


def _load_patch_config(model_dir: str | Path) -> LingBotPatchConfig:
    root = _transformer_root(model_dir)
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"LingBot transformer config does not exist: {config_path}"
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return LingBotPatchConfig(
        latent_channels=int(payload.get("in_channels", 16)),
        patch_size=tuple(int(value) for value in payload["patch_size"]),
        hidden_size=int(payload["hidden_size"]),
        num_attention_heads=int(payload["num_attention_heads"]),
        norm_eps=float(payload.get("norm_eps", 1e-6)),
        axes_dims=tuple(int(value) for value in payload["axes_dims"]),
        axes_lens=tuple(int(value) for value in payload["axes_lens"]),
        rope_theta=float(payload.get("rope_theta", 256.0)),
        patch_embed_bias=bool(payload.get("patch_embed_bias", True)),
    )


def _safetensor_for_key(root: Path, key: str) -> tuple[Path, str]:
    index_paths = sorted(root.glob("*.safetensors.index.json"))
    for index_path in index_paths:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map", {})
        matching = [name for name in weight_map if name.endswith(key)]
        if len(matching) == 1:
            name = matching[0]
            return root / weight_map[name], name
        if len(matching) > 1:
            raise RuntimeError(
                f"multiple tensors end with {key!r} in {index_path}: {matching}"
            )
    matches: list[tuple[Path, str]] = []
    for tensor_path in sorted(root.glob("*.safetensors")):
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            matching = [name for name in handle.keys() if name.endswith(key)]
        matches.extend((tensor_path, name) for name in matching)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one tensor ending with {key!r} below {root}, "
            f"found {matches}"
        )
    return matches[0]


def load_lingbot_patch_embedder(
    model_dir: str | Path,
) -> tuple[LingBotPatchConfig, nn.Linear]:
    """Load only LingBot's latent patch projection, never the DiT blocks."""

    root = _transformer_root(model_dir)
    config = _load_patch_config(model_dir)
    in_features = config.latent_channels * math.prod(config.patch_size)
    projection = nn.Linear(
        in_features,
        config.hidden_size,
        bias=config.patch_embed_bias,
    )
    weight_path, weight_key = _safetensor_for_key(
        root,
        "patch_embedder.weight",
    )
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(weight_key)
    if tuple(weight.shape) != tuple(projection.weight.shape):
        raise RuntimeError(
            "pretrained patch projection shape mismatch: "
            f"{tuple(weight.shape)} vs {tuple(projection.weight.shape)}"
        )
    projection.weight.data.copy_(weight.float())
    if projection.bias is not None:
        bias_path, bias_key = _safetensor_for_key(
            root,
            "patch_embedder.bias",
        )
        with safe_open(bias_path, framework="pt", device="cpu") as handle:
            bias = handle.get_tensor(bias_key)
        projection.bias.data.copy_(bias.float())
    return config, projection


class DecoderSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("decoder hidden size must be divisible by head count")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.output = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = tokens.shape
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
        return self.output(attended.transpose(1, 2).reshape(batch, length, hidden))


class LatentQueryDecoderBlock(nn.Module):
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
        self.attention = DecoderSelfAttention(hidden_size, num_heads)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.ffn(self.ffn_norm(tokens))


class PluckerRGBDecoder(nn.Module):
    """Target rays query scene memory and unpatchify directly to RGB."""

    def __init__(
        self,
        *,
        memory_hidden_size: int,
        image_height: int,
        image_width: int,
        patch_height: int,
        patch_width: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        norm_eps: float,
    ) -> None:
        super().__init__()
        if image_height % patch_height or image_width % patch_width:
            raise ValueError("decoder image size must be divisible by patch size")
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.gradient_checkpointing = False
        self.memory_projection = nn.Linear(memory_hidden_size, hidden_size)
        # Six Plucker channels plus the paper's binary visibility channel.
        # A decoder target is unobserved, so its visibility channel is zero.
        self.ray_patch_embedding = nn.Conv2d(
            7,
            hidden_size,
            kernel_size=(patch_height, patch_width),
            stride=(patch_height, patch_width),
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
            3 * patch_height * patch_width,
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
        grid_h = self.image_height // self.patch_height
        grid_w = self.image_width // self.patch_width
        return (
            patches.reshape(
                batch,
                grid_h,
                grid_w,
                self.patch_height,
                self.patch_width,
                3,
            )
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(batch, 3, self.image_height, self.image_width)
        )

    def decode_view(
        self,
        memory: torch.Tensor,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        import pdb; pdb.set_trace()
        memory_tokens = self.memory_projection(memory)
        queries = self._ray_queries(c2w, intrinsics)
        memory_length = memory_tokens.shape[1]
        tokens = torch.cat((memory_tokens, queries), dim=1)
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
        # Sigmoid fixes the RGB range while retaining dense gradients.
        return torch.sigmoid(self._unpatchify(self.output_projection(query_output)))

    def forward(
        self,
        memory: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        import pdb; pdb.set_trace()
        if target_c2w.ndim != 4 or target_c2w.shape[-2:] != (4, 4):
            raise ValueError("target_c2w must be [B,T,4,4]")
        if target_intrinsics.shape[:2] != target_c2w.shape[:2]:
            raise ValueError("target intrinsics must match target camera views")
        outputs = [
            self.decode_view(
                memory,
                target_c2w[:, view_index],
                target_intrinsics[:, view_index],
            )
            for view_index in range(target_c2w.shape[1])
        ]
        return torch.stack(outputs, dim=1)


class GIMMemoryReconstructionModel(nn.Module):
    """No-DiT experiment: VAE latents -> GIM memory -> novel-view RGB."""

    def __init__(
        self,
        patch_embedder: nn.Linear,
        config: MemoryReconstructionConfig,
    ) -> None:
        super().__init__()
        self.reconstruction_config = config
        patch_t, patch_h, patch_w = config.latent_patch_size
        if patch_t != 1:
            raise ValueError("independent-frame history requires patch_t=1")
        latent_h = config.image_height // config.vae_spatial_stride
        latent_w = config.image_width // config.vae_spatial_stride
        if latent_h % patch_h or latent_w % patch_w:
            raise ValueError("VAE latent resolution is not divisible by patch size")
        expected_in = config.latent_channels * math.prod(config.latent_patch_size)
        if (
            patch_embedder.in_features != expected_in
            or patch_embedder.out_features != config.encoder_hidden_size
        ):
            raise ValueError("patch embedder does not match reconstruction config")
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
        self.decoder = PluckerRGBDecoder(
            memory_hidden_size=config.encoder_hidden_size,
            image_height=config.image_height,
            image_width=config.image_width,
            patch_height=config.vae_spatial_stride * patch_h,
            patch_width=config.vae_spatial_stride * patch_w,
            hidden_size=config.decoder_hidden_size,
            depth=config.decoder_depth,
            num_heads=config.decoder_num_heads,
            mlp_ratio=config.decoder_mlp_ratio,
            norm_eps=config.decoder_norm_eps,
        )

    @property
    def patch_grid(self) -> tuple[int, int]:
        config = self.memory_encoder.config
        return config.patch_height, config.patch_width

    def patchify_history(self, latents: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = latents.shape
        patch_t, patch_h, patch_w = self.reconstruction_config.latent_patch_size
        if frames % patch_t or height % patch_h or width % patch_w:
            raise ValueError("history latent dimensions are not patch divisible")
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
        tokens = self.patchify_history(history_latents)
        import pdb; pdb.set_trace()
        cameras = camera_vector(
            history_c2w,
            history_intrinsics,
            (
                self.reconstruction_config.image_height,
                self.reconstruction_config.image_width,
            ),
        )
        return self.memory_encoder(tokens, cameras)

    def forward(
        self,
        history_latents: torch.Tensor,
        history_c2w: torch.Tensor,
        history_intrinsics: torch.Tensor,
        target_c2w: torch.Tensor,
        target_intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import pdb; pdb.set_trace()
        memory = self.build_memory(
            history_latents,
            history_c2w,
            history_intrinsics,
        )
        reconstructed = self.decoder(memory, target_c2w, target_intrinsics)
        return reconstructed, memory


def reconstruction_config_from_lingbot(
    patch: LingBotPatchConfig,
    **overrides: Any,
) -> MemoryReconstructionConfig:
    values: dict[str, Any] = {
        "latent_channels": patch.latent_channels,
        "latent_patch_size": patch.patch_size,
        "encoder_hidden_size": patch.hidden_size,
        "encoder_num_heads": patch.num_attention_heads,
        "encoder_norm_eps": patch.norm_eps,
        "encoder_axes_dims": patch.axes_dims,
        "encoder_axes_lens": patch.axes_lens,
        "encoder_rope_theta": patch.rope_theta,
    }
    values.update(overrides)
    return MemoryReconstructionConfig(**values)
