from __future__ import annotations

import torch

from lingbot_video.latent_spatial_memory.geometry import (
    intrinsics_vector_to_matrix,
    normalize_c2w_to_first_capture,
    resize_crop_intrinsics,
    scale_intrinsics,
)
from lingbot_video.latent_spatial_memory.memory import (
    LatentSpatialMemory,
    memory_consistency_mask,
)


def test_pose_normalization_sets_first_capture_to_identity() -> None:
    first = torch.eye(4)
    first[:3, 3] = torch.tensor([2.0, -1.0, 3.0])
    second = first.clone()
    second[:3, 3] += torch.tensor([1.0, 0.0, 0.0])
    normalized = normalize_c2w_to_first_capture(
        torch.stack((first, second)),
        first,
    )
    torch.testing.assert_close(normalized[0], torch.eye(4))
    torch.testing.assert_close(normalized[1, :3, 3], torch.tensor([1.0, 0.0, 0.0]))


def test_intrinsics_follow_resize_and_center_crop() -> None:
    vector = torch.tensor([1000.0, 1000.0, 960.0, 540.0])
    resized = resize_crop_intrinsics(vector, (1080, 1920), (480, 832))
    # 1920x1080 -> 853x480 then center-crop 10 pixels in x.
    assert resized[0] > 430
    assert resized[1] > 440
    assert 410 < resized[2] < 422
    assert 235 < resized[3] < 245


def test_same_view_memory_readout_recovers_latent_cells() -> None:
    height, width = 4, 6
    channels = 3
    latent = torch.arange(channels * height * width).reshape(channels, height, width).float()
    depth = torch.full((height, width), 2.0)
    intrinsics = intrinsics_vector_to_matrix(
        torch.tensor([4.0, 4.0, width / 2.0, height / 2.0])
    )
    c2w = torch.eye(4)
    memory = LatentSpatialMemory(
        channels,
        device="cpu",
        feature_dtype=torch.float32,
    )
    written = memory.write(
        latent,
        depth,
        intrinsics,
        c2w,
        image_hw=(height, width),
        relative_edge_threshold=0.0,
    )
    assert written == height * width
    readout = memory.read(
        c2w,
        scale_intrinsics(intrinsics, (height, width), (height, width)).unsqueeze(0),
        (height, width),
    )
    assert bool((readout.visibility == 1).all())
    torch.testing.assert_close(readout.features[:, 0], latent)


def test_same_depth_readout_prefers_latest_memory_update() -> None:
    height, width = 3, 4
    depth = torch.ones(height, width)
    intrinsics = torch.tensor(
        [[4.0, 0.0, 1.5], [0.0, 4.0, 1.0], [0.0, 0.0, 1.0]]
    )
    c2w = torch.eye(4)
    memory = LatentSpatialMemory(
        2,
        device=torch.device("cpu"),
        feature_dtype=torch.float32,
        voxel_size=0,
    )
    memory.write(
        torch.zeros(2, height, width),
        depth,
        intrinsics,
        c2w,
        image_hw=(height, width),
        frame_id=3,
    )
    latest = torch.ones(2, height, width)
    memory.write(
        latest,
        depth,
        intrinsics,
        c2w,
        image_hw=(height, width),
        frame_id=9,
    )

    readout = memory.read(c2w, intrinsics, (height, width))
    torch.testing.assert_close(readout.features[:, 0], latest)
    torch.testing.assert_close(readout.depth[0, 0], depth)


def test_memory_consistency_preserves_unseen_and_rejects_conflicts() -> None:
    candidate = torch.tensor([[2.0, 4.0, 3.0]])
    valid = torch.ones_like(candidate, dtype=torch.bool)
    projected = torch.tensor([[2.1, 2.0, 0.0]])
    visibility = torch.tensor([[1.0, 1.0, 0.0]])
    mask = memory_consistency_mask(
        candidate,
        valid,
        projected,
        visibility,
        relative_threshold=0.15,
    )
    torch.testing.assert_close(
        mask,
        torch.tensor([[True, False, True]]),
    )
