CUDA_VISIBLE_DEVICES=0 python scripts/inference_memory_3drae.py \
  --config configs/memory_3drae.json \
  --checkpoint /mnt/shared-storage-user/bianyichen/lingbot-video-output/memory-3drae/checkpoint-iter-00003200-step-00003200 \
  --item_name 9WS0D7j9yxc_024000_029000.mp4 \
  --output_dir /mnt/shared-storage-user/bianyichen/gim_inference/memory-3drae-iter3200
