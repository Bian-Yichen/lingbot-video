"""Active World Memory for long-horizon novel-view video generation.

The package keeps the original LingBot-Video backbone intact and implements a
hierarchical, iterative visual-memory policy around it.  Target RGB is never an
input to the retrieval policy: it is used only as a training target for the
world critic and the flow-matching generator objective.
"""

from .agent import ActiveMemoryRollout, ActiveWorldMemoryAgent
from .data import ActiveMemorySample, ActiveMemorySampleConfig, LocalRoomTourDataset
from .model import ActiveWorldMemoryConfig, ActiveWorldMemoryModel

__all__ = [
    "ActiveMemoryRollout",
    "ActiveMemorySample",
    "ActiveMemorySampleConfig",
    "ActiveWorldMemoryAgent",
    "ActiveWorldMemoryConfig",
    "ActiveWorldMemoryModel",
    "LocalRoomTourDataset",
]
