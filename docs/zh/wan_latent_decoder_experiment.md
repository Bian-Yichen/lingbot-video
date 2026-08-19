# Memory → Wan latent → Wan decoder 两阶段实验

本分支 `exp/memory-wan-latent-decoder` 从
`exp/memory-encoder-novel-view-decoder` 分出。它完全不加载或执行 LingBot DiT，目标是把
“memory encoder 是否可训练、是否保留了可解码场景信息”独立验证清楚。

数据路径是：

```text
不重叠 history RGB
  → 冻结 Wan VAE encoder（逐帧独立编码）
  → 冻结 LingBot latent patch projection
  → 现有两层 GIM compact memory encoder
  → camera Plücker-ray query latent renderer
  → 目标相机下的标准化 Wan latent
  → Wan VAE decoder
  → held-out novel-view RGB
```

target RGB 只作为监督信号。它不会进入 memory encoder，且 sampling 会把所有 target
frame 从 history candidates 中排除。默认同一份 memory 解码两个 held-out target views；
每个 target 都作为独立的一帧经过 Wan decoder，绝不会把不同相机 view 当成连续视频帧
塞进 Wan 的因果时间轴。

## 新架构

当前 GIM memory encoder 保持原样：Wan latent 以 `1×2×2` patch 投影到 1280 维，
history camera vector 广播到对应 patch，两个 compact-attention block 将任意数量 history
压缩成默认 `1×30×52=1560` 个 memory tokens。

新的 `PluckerWanLatentRenderer` 对每个目标相机生成 full-resolution 6-channel Plücker
rays 和一个全零 visibility channel。`16×16` ray patch embedding 产生 `30×52` 个
camera query tokens；它们与 memory tokens 一起经过默认 8 层、hidden 768、16 heads
的 transformer。最终只读取 query 部分，并预测 `16×2×2` 数值/patch，unpatchify 为
目标相机的 `[16,60,104]` 标准化 Wan latent。

相较直接预测 RGB，这样把任务拆成了两部分：renderer 负责从 scene memory + camera
恢复目标 latent；预训练 Wan decoder 负责将合法 latent 映射到自然图像。它与现有 Wan
encoder 的 latent 定义、通道统计和空间尺度完全对齐。

## 两阶段训练

总 loss 是：

```text
L = 1.0 * MSE(predicted standardized latent, target standardized latent)
  + 1.0 * MSE(decoded RGB, target RGB)
  + 0.1 * LPIPS(decoded RGB, target RGB)
```

阶段 1（默认 epoch 1–10）：

- 训练 GIM memory encoder 和 camera-query latent renderer；
- LingBot patch projection 默认冻结；
- Wan decoder 的全部预训练权重、LoRA 和 latent refiner 都处于恒等/关闭状态；
- RGB 与 LPIPS 的梯度仍穿过冻结 Wan decoder 回传到 predicted latent，不会在 decoder
  处 `detach`。

阶段 2（默认 epoch 11–20）：

- memory encoder 和 latent renderer 继续训练；
- 打开 Wan decoder mid-block attention 中 `to_qkv`、`proj` 的 1×1 Conv LoRA；
- 同时打开一个零初始化、全量训练的 native-latent residual refiner；
- Wan decoder 的原始卷积、残差块、上采样和输出层仍全部冻结。

adapter 参数在作业启动时就已经加入 optimizer/DDP 图；阶段 1 的 residual gate 与 adapter
learning rate 都为 0，阶段 2 再同时开启 gate 和独立 warmup。因此不需要在 DDP 运行中
动态增加参数，阶段边界前后的 checkpoint 都能直接续训。

Diffusers 0.37.1 的 Wan VAE 本身没有打开原生 gradient checkpointing。本分支利用
“每个 novel view 都是无跨帧状态的独立单帧”这一约束，在等价的无 temporal-cache
decoder 路径上对整个 Wan decoder 做 non-reentrant checkpoint；配置中的
`gradient_checkpointing=true` 会同时作用于 memory blocks、latent renderer 和这条
decoder 路径。

## History 慢启动

默认前 8 个 epoch 线性增大随机 history 范围：

| Epoch | 随机 history views |
|---:|---:|
| 1 | 2–4 |
| 2 | 3–8 |
| 3 | 5–12 |
| 4 | 6–16 |
| 5 | 8–20 |
| 6 | 9–24 |
| 7 | 11–28 |
| 8 及以后 | 12–32 |

每个 epoch 开始时训练脚本会真正替换 dataset sampling config，并用 epoch seed 重新采样；
不是只修改日志。checkpoint 保存精确的 next epoch、epoch 内 iteration 与 dataloader RNG，
所以中断续训不会重走错误的 curriculum。

## 安装和训练

LPIPS 依赖：

```bash
pip install -e '.[memory-reconstruction]'
```

单卡：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_wan_latent_reconstruction.py \
  --config configs/gim_wan_latent_decoder.json
```

多卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  scripts/train_wan_latent_reconstruction.py \
  --config configs/gim_wan_latent_decoder.json
```

本实验不加载 Qwen、text encoder 或 DiT，所以不需要
`LINGBOT_QWEN_ATTN_IMPLEMENTATION`。若先做单 scene overfit，建议覆盖
`--history_views_start_min 2 --history_views_start_max 2
--history_views_end_min 2 --history_views_end_max 2 --history_curriculum_epochs 1
--lr_warmup_steps 0 --decoder_warmup_steps 0 --lpips_weight 0`，先确认 latent MSE 和 RGB
MSE 能快速下降。

断点续训：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_wan_latent_reconstruction.py \
  --config configs/gim_wan_latent_decoder.json \
  --resume_from_checkpoint /path/to/checkpoint-iter-00000200-step-00000200
```

如果配置中 `save_optimizer_state=false`，模型、scheduler、阶段、curriculum 与数据位置会
恢复，但 Adam moments 会重新开始；希望严格无缝续训时改为 `true`。

## 推理诊断

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_wan_latent_reconstruction.py \
  --config configs/gim_wan_latent_decoder.json \
  --checkpoint /path/to/checkpoint-iter-00000200-step-00000200 \
  --item_name '<dataset scene directory name>' \
  --output_dir /tmp/gim-wan-latent-eval
```

`--stage auto` 默认恢复 checkpoint 记录的阶段，也可以显式指定 `--stage 1` 或
`--stage 2` 做 adapter ablation。输出包含：

- `*-prediction.png`：memory + camera 预测结果；
- `*-target.png`：原始监督图；
- `*-vae-reconstruction.png`：精确 target latent 经过未适配 Wan decoder 的重建上限；
- `latents.pt`：预测与 target 的标准化 latent；
- `metrics.json`：latent MSE、RGB MSE/PSNR、VAE reconstruction floor 和 memory norm。

若 `vae-reconstruction.png` 正常但 prediction 很差，问题在 memory/latent renderer；若
latent MSE 已经很低而 prediction 仍明显差，重点检查 decoder adapter 或 latent
normalization；若 target latent 的 VAE reconstruction 本身就差，则不能把这部分误判为
memory encoder 的失败。
