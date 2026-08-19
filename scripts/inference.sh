CUDA_VISIBLE_DEVICES=4 \
  python scripts/inference_geometry_aware_memory.py \
  --config configs/gim_world_local_inference.json \
  --checkpoint /mnt/shared-storage-user/bianyichen/lingbot-video-output/gim-flow-only/checkpoint-iter-00000600-step-00000600 \
  --item_name 9WS0D7j9yxc_024000_029000.mp4 \
  --output_dir /mnt/shared-storage-user/bianyichen/gim_inference/flow_only_iter600 \
  --local_window_start 798 \
  --target_start 818 \
  --memory_view_count 18 \
  --num_blocks 1 \
  --num_inference_steps 10 \
  --guidance_scale 1.0 \
  --seed 42