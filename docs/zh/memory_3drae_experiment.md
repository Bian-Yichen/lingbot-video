# Wan-3DRAE memory 实验

本分支 `exp/memory-3DRAE` 从 `exp/memory-wan-latent-decoder` 的
`c33ecb8` 分出。目标是保留 Wan VAE 前后端，但完整替换原来的 GIM
compact memory encoder 和 8 层 latent renderer：

```text
history RGB
  -> frozen Wan encoder（每个 view 独立编码）
  -> Wan latent patch projection
  -> 3DRAE 12-layer Latent Fuse Neck
  -> 1024 fixed-length scene tokens
  -> 3DRAE 16-layer target-ray query decoder
  -> target-view standardized Wan latent
  -> frozen / stage-2 adapted Wan decoder
  -> novel-view RGB
```

这里实现的是论文 memory 拓扑的 Wan-latent adaptation。论文原版使用冻结的
DINOv2-B/DA3/SigLIP2-B 作为 2D encoder，并直接从 query tokens
unpatchify RGB；本实验按既定 pipeline 把两端替换成 frozen Wan encoder 和
Wan decoder。论文可选的 point-map decoder 没有启用，因为当前数据/训练
schedule 不提供 point-map ground truth，且本实验要求保持数据逻辑不变。

## 与原 GIM 分支的结构差异

| 模块 | 父分支 | 本分支 |
|---|---|---|
| history token dim | LingBot patch embed，1280 | 新训练 Wan patch projection，768 |
| camera conditioning | view-level camera vector | patch-level 7-channel Pluecker ray + visibility |
| memory encoder | 2 个 GIM compact block + 3D RoPE | 12 层 global self-attention，无 RoPE |
| memory 数量 | 1560 | 固定 1024 |
| target decoder | 8 层 latent renderer | 16 层 global self-attention ray-query decoder |
| target 输出 | Wan latent | Wan latent |
| Wan decoder schedule | stage 1 frozen，stage 2 LoRA/refiner | 保持不变 |

3DRAE 不使用 GIM RoPE。history 和 target 的空间位置来自像素级 Pluecker
ray map，经与 Wan latent patch 相同网格的卷积投影后进入 transformer。

在 480×832 下，Wan latent 为 60×104；按 2×2 latent patch 切分后，每个
view 有 30×52=1560 tokens。encoder 序列长度为：

```text
1024 + history_views * 1560
```

decoder 对每个 target view 独立执行，单次序列长度为 `1024+1560=2584`；
memory 只构建一次，然后被所有 target camera 复用。

## 数据采样与 query curriculum

数据根目录、每 scene 每 epoch 一次采样、81-frame local window、pose-aware
facility-location history retrieval 均与父分支一致。

target 是 local window 中央一段连续帧。target 帧从 history candidates 中完全
剔除；候选 history 同时存在于 target 前后两侧，因此仍是：

```text
history candidates | held-out contiguous target | history candidates
```

facility-location 根据 pose 覆盖选择最终 history，所以最终入选数量来自配置的
min/max 区间；它不会让 target 与 history 重叠。

默认 8-epoch curriculum 同步增加 history 与 query：

| epoch（从 1 开始） | history min..max | 连续 query views |
|---:|---:|---:|
| 1 | 2..4 | 2 |
| 2 | 3..8 | 3 |
| 3 | 5..12 | 4 |
| 4 | 6..16 | 5 |
| 5 | 8..20 | 5 |
| 6 | 9..24 | 6 |
| 7 | 11..28 | 7 |
| 8+ | 12..32 | 8 |

可通过 `query_views_start/query_views_end/view_curriculum_epochs` 调整 query
增长范围。resume 使用 checkpoint 中的 `next_epoch` 和 dataloader RNG state，
不会重新开始 curriculum。

## 论文训练机制

- 1024 个 learnable scene queries 与全部 multi-view patch tokens 拼接。
- 12 层 global self-attention 融合后，只保留前 1024 个 scene tokens。
- memory 输出使用无 affine LayerNorm；SyncBatchNorm 在训练集上跨 worker 累积
  global running stats。论文使用预先在 ImageNet 计算的统计量，但这些统计量未
  随论文公开，且本分支的 latent 来源已改为 Wan，因此不能直接伪造或套用；
  checkpoint 会保存本实验实际估计出的 running stats。
- 每个 sample 以 0.1 概率执行 view masking；触发后随机隐藏 60%~90%
  history views，并用 visibility=0 的 ray tokens 保留 pose 信息。
- decoder 训练时给 memory 加噪：`sigma~Uniform(0,0.8)`，
  `memory'=memory+sigma*epsilon`；eval 自动关闭。
- 默认重建目标为 RGB MSE + LPIPS；Wan adaptation 独有的 latent MSE 默认权重
  为 0，可做消融时开启。
- DINOv2-Small hinge discriminator 从 step 50k 开始预热，adaptive GAN loss
  从 step 60k 开始加入 generator，权重 0.75。
- 父分支的两阶段 Wan decoder schedule 保持不变：前 10 epochs 冻结 Wan
  decoder；之后继续训练 3DRAE，同时启用 Wan mid-block attention LoRA 和
  zero-init latent refiner。

作者仓库当前只有 README，未公开训练代码。论文只说明 discriminator 从
DINOv2-Small 初始化，没有披露判别 head 和 GAN variant；本实现采用 DINOv2
CLS token + linear logit head、hinge GAN，并按 VAE/VQGAN 常用的 last-layer
gradient norm 计算 adaptive `omega_G`。这些是明确隔离的实现选择，不宣称为
作者未公开的代码细节。

## 训练

完整训练：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_memory_3drae.py \
  --config configs/memory_3drae.json
```

`LINGBOT_QWEN_ATTN_IMPLEMENTATION` 与本实验无关，因为本路径不加载 Qwen 或
LingBot DiT。3DRAE attention 使用 PyTorch scaled-dot-product attention；CUDA
环境会按 PyTorch 可用后端选择 flash / memory-efficient kernel。

如果训练机器不能在线读取 `facebook/dinov2-small`，先下载到本地并把
`discriminator_model_name_or_path` 改成本地目录。

只验证数据、Wan 和 3DRAE 前向/反向，不加载 LPIPS/GAN 的 smoke test：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_memory_3drae.py \
  --config configs/memory_3drae.json \
  --lpips_weight 0 \
  --gan_loss_weight 0 \
  --history_views_start_min 2 \
  --history_views_start_max 2 \
  --history_views_end_min 2 \
  --history_views_end_max 2 \
  --query_views_start 2 \
  --query_views_end 2 \
  --view_curriculum_epochs 1
```

### 显存/计算量警告

论文 fuse neck 是真正的 global self-attention，复杂度对 encoder token 数为
二次方。保持父分支最大 32 history views 时，encoder 序列达到 50,944 tokens；
即使 flash attention 降低显存，这一设置的计算量仍非常大。实现没有偷偷改成
local/chunked attention，因为那将不再是论文结构。建议先用 2~4 history 做
正确性和 overfit 验证，再根据实际 GPU 吞吐逐步提高；训练日志会打印最大
encoder token 数。

## 推理

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_memory_3drae.py \
  --config configs/memory_3drae.json \
  --checkpoint /path/to/checkpoint-iter-00000200-step-00000200 \
  --item_name 'zx8lnpzDG58_012000_017000.mp4' \
  --output_dir /tmp/memory-3drae-eval
```

固定同一个 local window、target 和 history 数量：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference_memory_3drae.py \
  --config configs/memory_3drae.json \
  --checkpoint /path/to/checkpoint \
  --item_name 'zx8lnpzDG58_012000_017000.mp4' \
  --local_window_start 500 \
  --target_start 536 \
  --memory_view_count 16 \
  --query_views 8 \
  --output_dir /tmp/memory-3drae-fixed
```

输出包含 prediction、target、exact-target-latent 的 Wan VAE reconstruction、
`latents_and_memory.pt` 和 `metrics.json`。metrics 额外记录 target 前后各有多少
入选 history view，以及 history/target overlap（必须为 0）。

父分支 checkpoint 与本分支结构不兼容，不能互相 resume。
