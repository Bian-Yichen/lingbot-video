from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .geometry import (
    backproject_depth,
    depth_validity_mask,
    downsample_depth_bilinear,
    scale_intrinsics,
)


@dataclass
class LatentMemoryReadout:
    features: torch.Tensor
    visibility: torch.Tensor
    depth: torch.Tensor


class LatentSpatialMemory:
    """Persistent 3D cache of native VAE latent tokens.

    The faithful Mirage representation is a union of ``(world point, latent
    feature)`` pairs.  Optional voxel consolidation and a point cap are exposed
    for multi-thousand-frame room tours, but are disabled unless configured.
    """

    def __init__(
        self,
        latent_channels: int,
        *,
        device: torch.device | str,
        feature_dtype: torch.dtype = torch.bfloat16,
        max_points: int = 0,
        voxel_size: float = 0.0,
    ) -> None:
        self.latent_channels = int(latent_channels)
        self.device = torch.device(device)
        self.feature_dtype = feature_dtype
        self.max_points = int(max_points)
        self.voxel_size = float(voxel_size)
        self.points = torch.empty(0, 3, device=self.device, dtype=torch.float32)
        self.features = torch.empty(
            0,
            self.latent_channels,
            device=self.device,
            dtype=self.feature_dtype,
        )
        self.confidence = torch.empty(0, device=self.device, dtype=torch.float32)
        self.frame_ids = torch.empty(0, device=self.device, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def clone(self) -> "LatentSpatialMemory":
        output = LatentSpatialMemory(
            self.latent_channels,
            device=self.device,
            feature_dtype=self.feature_dtype,
            max_points=self.max_points,
            voxel_size=self.voxel_size,
        )
        output.points = self.points.clone()
        output.features = self.features.clone()
        output.confidence = self.confidence.clone()
        output.frame_ids = self.frame_ids.clone()
        return output

    @torch.no_grad()
    def write(
        self,
        latent: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
        *,
        image_hw: Sequence[int],
        valid_mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
        frame_id: int = 0,
        min_depth: float = 0.1,
        max_depth: float = 20.0,
        relative_edge_threshold: float = 0.08,
        compact: bool = True,
    ) -> int:
        """Lift one clean latent frame into the cache using metric depth."""

        if latent.ndim != 3 or latent.shape[0] != self.latent_channels:
            raise ValueError(
                f"latent must be [{self.latent_channels},H,W], got {tuple(latent.shape)}"
            )
        if depth.ndim != 2:
            raise ValueError(f"depth must be [H,W], got {tuple(depth.shape)}")
        latent_h, latent_w = latent.shape[-2:]
        depth_latent = downsample_depth_bilinear(depth, (latent_h, latent_w))
        k_latent = scale_intrinsics(intrinsics, image_hw, (latent_h, latent_w))
        geometric_valid = depth_validity_mask(
            depth_latent,
            min_depth=min_depth,
            max_depth=max_depth,
            relative_edge_threshold=relative_edge_threshold,
        )
        if valid_mask is not None:
            if valid_mask.shape == (latent_h, latent_w):
                mask_latent = valid_mask.bool()
            else:
                mask_latent = torch.nn.functional.interpolate(
                    valid_mask.float()[None, None],
                    size=(latent_h, latent_w),
                    mode="nearest",
                )[0, 0].bool()
            geometric_valid &= mask_latent
        if not geometric_valid.any():
            return 0

        points_grid = backproject_depth(depth_latent, k_latent, c2w)
        features_grid = latent.permute(1, 2, 0)
        points = points_grid[geometric_valid].to(self.device, torch.float32)
        features = features_grid[geometric_valid].to(self.device, self.feature_dtype)
        if confidence is None:
            point_confidence = torch.ones(points.shape[0], device=self.device)
        else:
            confidence_latent = torch.nn.functional.interpolate(
                confidence.float()[None, None],
                size=(latent_h, latent_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            point_confidence = confidence_latent[geometric_valid].to(self.device)
        frame_ids = torch.full(
            (points.shape[0],),
            int(frame_id),
            device=self.device,
            dtype=torch.long,
        )
        self.points = torch.cat((self.points, points), dim=0)
        self.features = torch.cat((self.features, features), dim=0)
        self.confidence = torch.cat((self.confidence, point_confidence), dim=0)
        self.frame_ids = torch.cat((self.frame_ids, frame_ids), dim=0)
        if compact:
            self._compact_if_needed()
        return int(points.shape[0])

    @torch.no_grad()
    def write_video(
        self,
        latents: torch.Tensor,
        depths: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
        *,
        image_hw: Sequence[int],
        valid_masks: Optional[torch.Tensor] = None,
        frame_ids: Optional[torch.Tensor] = None,
        min_depth: float = 0.1,
        max_depth: float = 20.0,
        relative_edge_threshold: float = 0.08,
    ) -> int:
        """Write ``T`` clean latent frames; tensors are [C,T,H,W]/[T,H,W]."""

        if latents.ndim != 4:
            raise ValueError(f"latents must be [C,T,H,W], got {tuple(latents.shape)}")
        total = 0
        for index in range(latents.shape[1]):
            total += self.write(
                latents[:, index],
                depths[index],
                intrinsics[index],
                c2w[index],
                image_hw=image_hw,
                valid_mask=None if valid_masks is None else valid_masks[index],
                frame_id=index if frame_ids is None else int(frame_ids[index].item()),
                min_depth=min_depth,
                max_depth=max_depth,
                relative_edge_threshold=relative_edge_threshold,
                compact=False,
            )
        self._compact_if_needed()
        return total

    @torch.no_grad()
    def compact(self) -> None:
        """Apply optional voxel consolidation and the configured point cap."""

        self._compact_if_needed()

    @torch.no_grad()
    def read(
        self,
        c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        output_hw: Sequence[int],
    ) -> LatentMemoryReadout:
        """Project memory with a frontmost-point z-buffer at latent resolution."""

        output_h, output_w = (int(value) for value in output_hw)
        if c2w.ndim == 2:
            c2w = c2w.unsqueeze(0)
        if intrinsics.ndim == 2:
            intrinsics = intrinsics.unsqueeze(0)
        if c2w.shape[0] != intrinsics.shape[0]:
            raise ValueError("target pose and intrinsics frame counts differ")
        num_views = c2w.shape[0]
        output_features = torch.zeros(
            self.latent_channels,
            num_views,
            output_h,
            output_w,
            device=self.device,
            dtype=self.feature_dtype,
        )
        output_visibility = torch.zeros(
            1,
            num_views,
            output_h,
            output_w,
            device=self.device,
            dtype=torch.float32,
        )
        output_depth = torch.zeros_like(output_visibility)
        if len(self) == 0:
            return LatentMemoryReadout(output_features, output_visibility, output_depth)

        homogeneous = torch.cat(
            (self.points, torch.ones(len(self), 1, device=self.device)),
            dim=1,
        )
        point_indices = torch.arange(len(self), device=self.device, dtype=torch.long)
        for view_index in range(num_views):
            w2c = torch.linalg.inv(c2w[view_index].float())
            camera = homogeneous @ w2c.T
            xyz = camera[:, :3]
            z = xyz[:, 2]
            projected = xyz @ intrinsics[view_index].float().T
            u = torch.floor(projected[:, 0] / z.clamp_min(1e-8)).long()
            v = torch.floor(projected[:, 1] / z.clamp_min(1e-8)).long()
            valid = (
                torch.isfinite(xyz).all(dim=1)
                & (z > 0)
                & (u >= 0)
                & (u < output_w)
                & (v >= 0)
                & (v < output_h)
            )
            if not valid.any():
                continue
            u_valid = u[valid]
            v_valid = v[valid]
            z_valid = z[valid]
            source_indices = point_indices[valid]
            linear = v_valid * output_w + u_valid
            cell_count = output_h * output_w
            min_depth = torch.full(
                (cell_count,),
                float("inf"),
                device=self.device,
                dtype=torch.float32,
            )
            min_depth.scatter_reduce_(0, linear, z_valid, reduce="amin", include_self=True)
            winners = torch.isclose(
                z_valid,
                min_depth[linear],
                rtol=1e-5,
                atol=1e-6,
            )
            winner_cells = linear[winners]
            winner_sources = source_indices[winners]
            newest_frame = torch.full(
                (cell_count,),
                torch.iinfo(torch.long).min,
                device=self.device,
                dtype=torch.long,
            )
            newest_frame.scatter_reduce_(
                0,
                winner_cells,
                self.frame_ids[winner_sources],
                reduce="amax",
                include_self=True,
            )
            newest = (
                self.frame_ids[winner_sources]
                == newest_frame[winner_cells]
            )
            newest_cells = winner_cells[newest]
            newest_sources = winner_sources[newest]
            best_confidence = torch.full(
                (cell_count,),
                -float("inf"),
                device=self.device,
                dtype=torch.float32,
            )
            best_confidence.scatter_reduce_(
                0,
                newest_cells,
                self.confidence[newest_sources],
                reduce="amax",
                include_self=True,
            )
            best = torch.isclose(
                self.confidence[newest_sources],
                best_confidence[newest_cells],
            )
            winner_index = torch.full(
                (cell_count,),
                len(self),
                device=self.device,
                dtype=torch.long,
            )
            winner_index.scatter_reduce_(
                0,
                newest_cells[best],
                newest_sources[best],
                reduce="amin",
                include_self=True,
            )
            occupied = winner_index < len(self)
            if not occupied.any():
                continue
            flat_features = output_features[:, view_index].reshape(self.latent_channels, -1)
            flat_visibility = output_visibility[0, view_index].reshape(-1)
            flat_depth = output_depth[0, view_index].reshape(-1)
            flat_features[:, occupied] = self.features[winner_index[occupied]].T
            flat_visibility[occupied] = 1.0
            flat_depth[occupied] = min_depth[occupied]
        return LatentMemoryReadout(output_features, output_visibility, output_depth)

    @torch.no_grad()
    def _compact_if_needed(self) -> None:
        if self.voxel_size > 0 and len(self) > 0:
            voxel = torch.floor(self.points / self.voxel_size).to(torch.int64)
            _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
            count = int(inverse.max().item()) + 1
            # Preserve the newest/highest-confidence observation in each voxel
            # rather than averaging latent tokens off their native manifold.
            max_frame = torch.full(
                (count,),
                torch.iinfo(torch.long).min,
                device=self.device,
                dtype=torch.long,
            )
            max_frame.scatter_reduce_(
                0,
                inverse,
                self.frame_ids,
                reduce="amax",
                include_self=True,
            )
            newest = self.frame_ids == max_frame[inverse]
            candidate_confidence = torch.where(
                newest,
                self.confidence,
                torch.full_like(self.confidence, -float("inf")),
            )
            max_confidence = torch.full(
                (count,),
                -float("inf"),
                device=self.device,
            )
            max_confidence.scatter_reduce_(
                0,
                inverse,
                candidate_confidence,
                reduce="amax",
                include_self=True,
            )
            candidates = newest & torch.isclose(
                self.confidence,
                max_confidence[inverse],
            )
            indices = torch.arange(len(self), device=self.device)
            selected = torch.full((count,), len(self), device=self.device, dtype=torch.long)
            selected.scatter_reduce_(
                0,
                inverse[candidates],
                indices[candidates],
                reduce="amin",
                include_self=True,
            )
            selected = selected[selected < len(self)]
            self._select(selected)
        if self.max_points > 0 and len(self) > self.max_points:
            # Uniformly retain the full trajectory instead of only the latest
            # frames, which is critical for closed-loop revisits.
            selected = torch.linspace(
                0,
                len(self) - 1,
                self.max_points,
                device=self.device,
            ).round().long()
            self._select(selected)

    def _select(self, selected: torch.Tensor) -> None:
        self.points = self.points[selected]
        self.features = self.features[selected]
        self.confidence = self.confidence[selected]
        self.frame_ids = self.frame_ids[selected]


def memory_consistency_mask(
    candidate_depth: torch.Tensor,
    candidate_valid: torch.Tensor,
    projected_depth: torch.Tensor,
    visibility: torch.Tensor,
    *,
    relative_threshold: float,
) -> torch.Tensor:
    """Reject writes that disagree with geometry already stored in memory.

    Unobserved cells remain writable so generation can expand the scene.  This
    is the no-external-model replacement for Mirage's semantic dynamic-object
    filter and also suppresses large VIPE/predicted-depth outliers.
    """

    if not (
        candidate_depth.shape
        == candidate_valid.shape
        == projected_depth.shape
        == visibility.shape
    ):
        raise ValueError("all memory consistency tensors must have equal shapes")
    if relative_threshold <= 0:
        return candidate_valid.bool()
    observed = visibility > 0.5
    denominator = torch.maximum(
        candidate_depth.abs(),
        projected_depth.abs(),
    ).clamp_min(1e-6)
    relative_error = (candidate_depth - projected_depth).abs() / denominator
    consistent = torch.isfinite(relative_error) & (
        relative_error <= relative_threshold
    )
    return candidate_valid.bool() & (~observed | consistent)
