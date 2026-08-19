from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from lingbot_video.latent_spatial_memory.data import (
    RGB_SAMPLE_KEYS,
    LocalRoomTourIndex,
    VipeRoomTourItem,
    normalize_preloaded_rgb,
)


def test_local_room_tour_index_lists_and_resolves_direct_children(
    tmp_path: Path,
) -> None:
    (tmp_path / "scene_b").mkdir()
    (tmp_path / "scene_a").mkdir()
    (tmp_path / "not_a_scene.txt").write_text("ignored", encoding="utf-8")

    index = LocalRoomTourIndex(tmp_path)

    assert index.list_items() == ["scene_a", "scene_b"]
    assert index.item_path("scene_a/") == (tmp_path / "scene_a").resolve()
    with pytest.raises(ValueError, match="direct child"):
        index.item_path("../scene_a")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        index.item_path("missing")


def test_worker_rgb_stays_uint8_until_compute_device_normalization(
    tmp_path: Path,
) -> None:
    rgb_path = tmp_path / "000000.png"
    Image.new("RGB", (4, 2), color=(0, 127, 255)).save(rgb_path)
    item = VipeRoomTourItem.__new__(VipeRoomTourItem)
    item.rgb_by_index = {0: rgb_path}

    rgb = item.read_rgb(0, (2, 4))

    assert rgb.dtype == torch.uint8
    assert tuple(rgb.shape) == (3, 2, 4)
    batch = {key: rgb.clone() for key in RGB_SAMPLE_KEYS}
    normalize_preloaded_rgb(batch)
    expected = torch.tensor([0.0, 127.0 / 255.0, 1.0])
    for value in batch.values():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value[:, 0, 0], expected)
