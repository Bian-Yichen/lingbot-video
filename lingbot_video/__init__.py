from __future__ import annotations

import importlib
from typing import Any

from .compat import install_torch26_custom_op_compat


# This runs before any lazy import of Diffusers/Transformers-backed modules.  It
# is a no-op on newer PyTorch versions and fixes postponed type annotations in
# torch.library custom-op schemas on PyTorch 2.6 and older.
install_torch26_custom_op_compat()


_EXPORTS = {
    "ActiveWorldMemoryConfig": (
        "lingbot_video.active_world_memory.model",
        "ActiveWorldMemoryConfig",
    ),
    "ActiveWorldMemoryModel": (
        "lingbot_video.active_world_memory.model",
        "ActiveWorldMemoryModel",
    ),
    "FlowUniPCMultistepScheduler": (
        "lingbot_video.scheduling_flow_unipc",
        "FlowUniPCMultistepScheduler",
    ),
    "LingBotVideoImageToVideoPipeline": (
        "lingbot_video.pipeline_lingbot_video_i2v",
        "LingBotVideoImageToVideoPipeline",
    ),
    "LingBotVideoPipeline": (
        "lingbot_video.pipeline_lingbot_video",
        "LingBotVideoPipeline",
    ),
    "LingBotVideoTransformer3DModel": (
        "lingbot_video.transformer_lingbot_video",
        "LingBotVideoTransformer3DModel",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(importlib.import_module(module_name), attr_name)
    globals()[name] = value
    return value
