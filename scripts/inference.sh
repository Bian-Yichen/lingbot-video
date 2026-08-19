CUDA_VISIBLE_DEVICES=4 \
  python scripts/inference_geometry_aware_memory.py \
  --config configs/gim_world_local_inference.json \
  --checkpoint /mnt/shared-storage-user/bianyichen/lingbot-video-output/gim-single-image-memory/checkpoint-iter-00001400-step-00001400 \
  --item_name 9WS0D7j9yxc_024000_029000.mp4 \
  --output_dir /mnt/shared-storage-user/bianyichen/gim_inference/single_image_memory_iter1400 \
  --local_window_start 818 \
  --target_start 818 \
  --memory_view_count 1 \
  --num_blocks 1 \
  --num_inference_steps 40 \
  --guidance_scale 1.0 \
  --seed 42