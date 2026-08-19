# 无 DiT 的 Memory Novel-View Reconstruction 实验

本分支从 `exp/flow-matching-only` 分出，目的只有一个：验证现有 GIM memory
encoder 产生的固定长度 memory 是否包含足够信息，可以在完全不经过 LingBot DiT
的情况下被相机条件 decoder 还原为图像。

## 论文结论与本实验选择

《Any 3D Scene is Worth 1K Tokens》的第一阶段 3DRAE 并不训练 DiT。它把若干
observed views 编码成固定长度 3D latent tokens，再用目标相机的 Plücker ray-map
tokens 查询这些 latent，直接重建 held-out novel-view RGB。论文同时训练 fuse neck
和 decoder，冻结前面的 2D representation encoder。

论文一次 scene sample 使用多视角数据，decoder 数学定义允许任意数量目标视角；
每个目标视角只输入 camera rays，不输入其 RGB。训练明确只用 novel-view rendering
监督。因此不能把某张 history frame 同时当 target：那主要验证复制输入，不验证
view-decoupled scene memory。

当前默认设置为：

- 从局部 81 帧窗口中选择 12–32 张、与 target 不重叠的 history views；
- VAE 只编码 history，target RGB 不经过 VAE，也不进入 memory；
- 同一份 memory 解码 2 张 held-out novel views；
- 每张 target 由自己的 Plücker ray map 查询 memory，decoder 内部逐视角执行并共享权重；
- inference 可按相同方式顺序解码任意数量的新相机视角。

## 当前 encoder 与 3DRAE encoder 并不相同

| 部分 | 论文 3DRAE | 本实验保留的现有 GIM encoder |
|---|---|---|
| 图像特征 | 冻结 DINOv2/DA3/SigLIP2 ViT-Base | 冻结 Wan VAE + LingBot latent patch projection |
| 相机注入 | 每个 patch 的 Plücker ray-map embedding | 每个 view 一个 16D camera vector，广播到其 patch tokens |
| 多视角融合 | 12 层 global self-attention fuse neck | 2 层 compact-attention GIM memory blocks |
| memory 长度 | 256/512/1024 tokens | 默认 `1×30×52=1560` tokens |
| 位置处理 | ray map 提供空间位置 | history/memory compact attention 使用 3D RoPE |

所以本分支不是把论文 encoder 冒充成现有 encoder，而是一个受控的可解码性实验：
只替换下游 DiT，保留现有 encoder。若该实验能够收敛，说明现有 memory 本身可训练、
之前的问题更可能在 DiT 读取 memory 的路径；若训练集 novel-view reconstruction 仍然
学不会，才有充分证据怀疑 encoder 容量、相机注入或压缩方式。

## Decoder 与 loss

`PluckerRGBDecoder` 按论文的 latent-query 思路实现：

1. 在目标分辨率生成 6-channel Plücker rays，并附加全零 visibility channel；
2. 使用 `16×16` stride/kernel patch embedding 得到 `30×52` ray query tokens；
3. 将 ray queries 与投影后的 memory tokens 拼接；
4. 默认经过 16 层、hidden size 768、16 heads 的 global self-attention transformer；
5. 只取最终 ray tokens，线性投影并 unpatchify 成 `480×832` RGB。

训练目标为：

```text
L = MSE(predicted_rgb, target_rgb) + 1.0 * LPIPS(predicted_rgb, target_rgb)
```

论文还使用延迟开启的 adversarial loss，并可选 VGGT point-map loss。本实验第一轮明确
关闭这两项：GAN 会引入新的 discriminator 优化变量，point-map 又会重新引入要消融的
geometry/VGGT 监督。论文自身也报告了无 adversarial loss 的有效 ablation；等 RGB
重建基线能稳定收敛后，再单独增加 GAN 才有可解释性。

## 安装与训练

LPIPS 是本实验新增的可选依赖：

```bash
pip install -e '.[memory-reconstruction]'
```

单卡训练：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_memory_reconstruction.py \
  --config configs/gim_memory_novel_view_decoder.json
```

多卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  scripts/train_memory_reconstruction.py \
  --config configs/gim_memory_novel_view_decoder.json
```

这里不加载 text encoder、Qwen 或 DiT，因此不需要设置
`LINGBOT_QWEN_ATTN_IMPLEMENTATION`。PyTorch SDPA 会自动使用当前环境可用的高效
CUDA attention kernel。

如果第一轮显存不足，可先用 `--decoder_depth 8 --memory_views_max 16` 做 smoke
test；单 scene overfit 时再加 `--lr_warmup_steps 0`，避免论文默认的 8000-step
warmup 让极短实验看起来没有学习。不要让 target 与 history 重叠，也不要把 target
RGB 输入 encoder。

## Checkpoint 推理

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_memory_reconstruction.py \
  --config configs/gim_memory_novel_view_decoder.json \
  --checkpoint /path/to/checkpoint-iter-00000200-step-00000200 \
  --item_name '<dataset scene directory name>' \
  --output_dir /tmp/gim-memory-reconstruction-eval
```

脚本会保存每个 held-out frame 的 `prediction.png`、`target.png` 和 `metrics.json`。
要完全复现实验采样，可同时指定内部时间轴上的
`--local_window_start`、`--target_start`、`--memory_view_count`；三者必须一起给出。
源 RGB 文件 index 等于内部 index 乘以 5。

## 最小判据

先只用极少数 scene 做 overfit：如果 memory encoder 与 decoder 的梯度都非零，且训练
PSNR 明显上升、LPIPS 下降，说明路径可训练。只有 overfit 通过后，才值得跑完整数据集
并比较不同 history view 数量与 novel-view camera 距离。
