#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 2
fi

"$PYTHON_BIN" -m pip install -U pip setuptools wheel --timeout "$PIP_TIMEOUT"

# PyTorch 2.6 is the newest official cu118 build that still has Linux wheels
# compatible with glibc 2.17.  The local LingBot compatibility shim handles the
# postponed-annotation custom-op import issue in newer Diffusers/Transformers.
"$PYTHON_BIN" -m pip install \
  'torch==2.6.0+cu118' \
  'torchvision==0.21.0+cu118' \
  --index-url https://download.pytorch.org/whl/cu118 \
  --timeout "$PIP_TIMEOUT"

"$PYTHON_BIN" -m pip install \
  -r requirements-glibc217-cu118.txt \
  --timeout "$PIP_TIMEOUT"

# Dependencies are installed explicitly above.  --no-deps prevents pyproject
# resolution from replacing the tested compatibility versions.
"$PYTHON_BIN" -m pip install -e . --no-deps

"$PYTHON_BIN" - <<'PY'
import torch
import torchvision
import diffusers
import transformers
import lingbot_video
from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("LingBotVideoPipeline import: OK")
PY
