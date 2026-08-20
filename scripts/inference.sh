CUDA_VISIBLE_DEVICES=6 python scripts/inference_wan_latent_reconstruction.py \
  --config configs/gim_wan_latent_decoder.json \
  --checkpoint /mnt/shared-storage-user/bianyichen/lingbot-video-output/gim-wan-latent-decoder-output/checkpoint-iter-00018200-step-00018200 \
  --item_name 9WS0D7j9yxc_024000_029000.mp4 \
  --output_dir /mnt/shared-storage-user/bianyichen/gim_inference/gim-wan-latent-decoder-iter18200