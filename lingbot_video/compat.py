from __future__ import annotations

"""Compatibility helpers for older PyTorch runtimes.

LingBot-Video's reference environment uses a recent PyTorch nightly.  PyTorch
2.6's custom-op schema inference does not resolve postponed/string type
annotations before validating a function signature.  Recent Diffusers and
Transformers define several custom ops in modules that use
``from __future__ import annotations``; importing those modules therefore fails
with an ``unsupported type torch.Tensor`` error on PyTorch 2.6.

The patch below is intentionally narrow: it only resolves a decorated
function's annotations immediately before PyTorch infers its custom-op schema.
It does not alter operator execution, dispatch, kernels, or model numerics.
"""

import importlib
import typing
from types import ModuleType
from typing import Any, Callable


def _torch_version_tuple(torch_module: ModuleType) -> tuple[int, int]:
    raw = str(getattr(torch_module, "__version__", "0.0")).split("+", 1)[0]
    parts = raw.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def _resolve_annotations(fn: Callable[..., Any]) -> dict[str, Any] | None:
    annotations = getattr(fn, "__annotations__", None)
    if not isinstance(annotations, dict):
        return None
    if not any(isinstance(value, str) for value in annotations.values()):
        return None

    globalns = getattr(fn, "__globals__", None)
    if not isinstance(globalns, dict):
        globalns = {}
    try:
        return typing.get_type_hints(fn, globalns=globalns, localns=globalns)
    except Exception:
        # Keep the original PyTorch error if an annotation genuinely cannot be
        # resolved.  Silently guessing a schema would be unsafe.
        return None


def _patch_infer_schema(module: ModuleType) -> bool:
    original = getattr(module, "infer_schema", None)
    if original is None or getattr(original, "_lingbot_string_annotation_compat", False):
        return False

    def infer_schema_compat(fn: Callable[..., Any], mutates_args: Any):
        original_annotations = getattr(fn, "__annotations__", None)
        resolved = _resolve_annotations(fn)
        if resolved is None:
            return original(fn, mutates_args)

        fn.__annotations__ = resolved
        try:
            return original(fn, mutates_args)
        finally:
            if original_annotations is None:
                try:
                    del fn.__annotations__
                except AttributeError:
                    pass
            else:
                fn.__annotations__ = original_annotations

    infer_schema_compat._lingbot_string_annotation_compat = True  # type: ignore[attr-defined]
    infer_schema_compat.__wrapped__ = original  # type: ignore[attr-defined]
    module.infer_schema = infer_schema_compat
    return True


def install_torch26_custom_op_compat() -> bool:
    """Install the postponed-annotation schema patch on PyTorch <= 2.6.

    Returns ``True`` when at least one internal schema-inference entry point was
    patched.  On newer PyTorch versions this function is a no-op.
    """

    try:
        import torch
    except Exception:
        return False

    if _torch_version_tuple(torch) > (2, 6):
        return False

    patched = False
    for module_name in ("torch._custom_op.impl", "torch._library.infer_schema"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        patched = _patch_infer_schema(module) or patched
    return patched
