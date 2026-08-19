#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-configs/latent_spatial_memory_stage1.json}"
MODEL_DIR="${MODEL_DIR:-/mnt/shared-storage-user/bianyichen/lingbot-video-dense-1.3b}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

accelerate launch --num_processes "$NUM_PROCESSES" \
  scripts/train_latent_spatial_memory.py \
  --config "$CONFIG" \
  --model_dir "$MODEL_DIR" \
  $EXTRA_ARGS
