#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

accelerate launch --config_file /mnt/petrelfs/bianyichen/.cache/huggingface/accelerate/single_gpu.yaml \
  scripts/train_latent_spatial_memory.py \
  --config configs/latent_spatial_memory_stage1.json \
  --model_dir /mnt/petrelfs/bianyichen/.cache/huggingface/hub/models--robbyant--lingbot-video-dense-1.3b/snapshots/f9789a7d9b4772a47aba62d4eb5282ddefd1da21
