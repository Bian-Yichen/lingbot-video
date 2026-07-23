from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class LongSceneConfig:
    """Configuration for recurrent, ray-addressed long-scene training.

    Capture frames are never selected by a target-pose retriever.  Every valid
    frame is streamed through a fixed-size fast state and periodically
    consolidated into a persistent slow state.  Generated frames use the same
    writer and remain branch-local until explicitly verified.
    """

    capture_chunk_size: int = 16
    slow_memory_tokens: int = 512
    fast_memory_tokens: int = 128
    memory_width: int = 768
    memory_heads: int = 12
    memory_update_layers: int = 2
    memory_consolidation_layers: int = 2
    memory_input_grid_h: int = 8
    memory_input_grid_w: int = 8
    consolidation_interval: int = 8
    truncate_memory_bptt_every: int = 8
    memory_confidence_decay: float = 0.999
    generated_slow_write_scale: float = 0.1
    verified_slow_write_scale: float = 1.0
    detach_generated_writes: bool = True

    ray_fourier_bands: int = 6
    geometry_feature_dim: int = 0
    max_geometry_queries: int = 512
    max_memory_reconstruction_queries: int = 512

    prefix_latent_frames: int = 1
    flow_loss_weight: float = 1.0
    rollout_flow_loss_weight: float = 0.5
    memory_state_consistency_loss_weight: float = 0.05
    memory_reconstruction_loss_weight: float = 0.05
    geometry_depth_loss_weight: float = 0.05
    geometry_feature_loss_weight: float = 0.05
    reprojection_loss_weight: float = 0.1
    paired_trajectory_loss_weight: float = 0.1
    geometry_depth_scale_weight: float = 0.1
    geometry_feature_scale_weight: float = 0.1
    occlusion_relative_threshold: float = 0.05
    occlusion_absolute_threshold: float = 0.01

    sigma_logit_mean: float = 0.0
    sigma_logit_std: float = 1.0
    geometry_loss_max_sigma: float = 0.7
    depth_is_ray_distance: bool = False

    def __post_init__(self) -> None:
        positive_integers = {
            "capture_chunk_size": self.capture_chunk_size,
            "slow_memory_tokens": self.slow_memory_tokens,
            "fast_memory_tokens": self.fast_memory_tokens,
            "memory_width": self.memory_width,
            "memory_heads": self.memory_heads,
            "memory_update_layers": self.memory_update_layers,
            "memory_consolidation_layers": self.memory_consolidation_layers,
            "memory_input_grid_h": self.memory_input_grid_h,
            "memory_input_grid_w": self.memory_input_grid_w,
            "consolidation_interval": self.consolidation_interval,
            "max_geometry_queries": self.max_geometry_queries,
            "max_memory_reconstruction_queries": (
                self.max_memory_reconstruction_queries
            ),
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise ValueError(f"These fields must be positive: {invalid}")
        if self.truncate_memory_bptt_every < 0:
            raise ValueError("truncate_memory_bptt_every must be non-negative")
        if self.memory_width % self.memory_heads != 0:
            raise ValueError("memory_width must be divisible by memory_heads")
        if not 0.0 < self.memory_confidence_decay <= 1.0:
            raise ValueError("memory_confidence_decay must be in (0, 1]")
        if not 0.0 <= self.generated_slow_write_scale <= 1.0:
            raise ValueError("generated_slow_write_scale must be in [0, 1]")
        if not 0.0 <= self.verified_slow_write_scale <= 1.0:
            raise ValueError("verified_slow_write_scale must be in [0, 1]")
        non_negative = {
            name: getattr(self, name)
            for name in (
                "flow_loss_weight",
                "rollout_flow_loss_weight",
                "memory_state_consistency_loss_weight",
                "memory_reconstruction_loss_weight",
                "geometry_depth_loss_weight",
                "geometry_feature_loss_weight",
                "reprojection_loss_weight",
                "paired_trajectory_loss_weight",
                "geometry_depth_scale_weight",
                "geometry_feature_scale_weight",
                "occlusion_relative_threshold",
                "occlusion_absolute_threshold",
            )
        }
        invalid = [name for name, value in non_negative.items() if value < 0.0]
        if invalid:
            raise ValueError(f"These fields must be non-negative: {invalid}")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "LongSceneConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown LongSceneConfig fields: {unknown}")
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "LongSceneConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")
