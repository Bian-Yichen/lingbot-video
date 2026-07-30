# GIM-World × LingBot-Video 训练与推理

本分支从 `compat/glibc217-torch26` 独立建立，不包含 MIRAGE 的 ControlNet、显式点云 memory 或 capture retrieval。实现对应论文 *Geometry-Aware Implicit Memory for Video World Models*（GIM-World，arXiv:2606.02436v1）。

## 数据时间轴

数据直接从机器上已经挂载好的本地目录读取，例如：

```text
/mnt/datasets/AnyReconProDataset_labeled_2/
```

`dataset_root` 的直接子目录就是 scene 条目。loader 不调用 rclone，也不会把
RGB、pose、intrinsics 或 metadata 复制到 `/tmp`；训练 worker 直接打开
`dataset_root/<item_name>/...`。每个条目的基本结构为：

```text
<dataset_root>/<item_name>/
├── RGB/
├── chunk_metadata.json
└── vipe/vipe_artifacts/
    ├── pose/video.npz
    └── intrinsics/video.npz
```

ViPE 的 RGB、pose 和 intrinsics 都只保留源视频的 `0,5,10,...` 帧；loader
首先把它们归一成内部连续时间轴 `0,1,2,...`。例如，一个 5000 帧源视频
条目会成为约 1000 帧训练序列。

每个 scene 第一次出现时，代码按 Wan VAE 官方 `_encode` 的时序调度运行：
首帧单独进入 encoder，之后每 4 帧一组；所有 causal convolution 的
`_enc_feat_map` 在整条约 1000 帧序列上持续更新。`81` 只是一次从磁盘预读的
RGB 数量，不会重置 VAE 状态。最终得到约 250 个 latent frames，写入
`latent_cache_root`。后续对同一 scene 的多个 iteration 直接复用该缓存。
缓存 metadata 带有 `wan_official_feature_cache_v1` 版本，旧的独立 clip
编码缓存会自动失效并重建。

监督 target 不从上述整场 latent 中硬切。每个 81-RGB target 都从空的 Wan
causal state 独立编码成 21 latent frames，保证其首 latent 语义与推理时独立
解码一致；结果按 target start 存入 `latent_cache_root/<scene>/target_clips/`，
下次抽到同一 window 时直接复用。默认至少 800 帧 history 后才取 target，
一个 1000 帧条目最多只有约 120 个 target cache（fp16 下约 0.5 GB/scene）；
磁盘紧张时可以安全删除该目录，训练会按需重建。

默认一个 iteration：

1. 从满足最少 800 帧历史的位置中选择连续 81 张 RGB 作为监督目标，对应 21 个 VAE latent frames。
2. 默认 `context_policy=prefix`，target 之前的全部 observation 都是 memory；在 1000 帧条目中每次约有 800–919 张历史 RGB，即约 200–230 个历史 latent frames。
3. 所有候选历史 latent 都进入论文的 pose-time GP mutual-information greedy selection，保留 `K=200`。这不是 frame retrieval：目标 camera pose 不参与选择，选择只压缩完整 capture history。
4. 被选中的 latent 按原时间顺序进入 memory encoder。一个 scene 默认连续产生 16 个不同 target window 的 iteration，从而复用已经建立的整场 VAE latent cache。

这与论文的因果 rollout 数据协议一致，也从机制上避免监督 RGB 通过未来的
causal VAE receptive field 泄漏进 memory。`all_except_target` 仍作为离线
完整-capture ablation 保留，但强制 `target_guard_rgb_frames >= 128`；guard
会同时排除目标前后邻域，因此可用候选可能少于 `K=200`。

## 模型路径

历史 latent 的形状在默认 480×832 分辨率下为：

```text
[B, 16, T_history, 60, 104]
  -> LingBot patch embedding (1×2×2)
[B, T_history, 30×52, 2048]
```

每个历史 frame 的 camera-to-world 3×4 矩阵和归一化 `fx,fy,cx,cy` 组成 16 维向量，经一个 linear embedding 加到该 frame 的所有 patch tokens。20 个 pose-free learnable query grids 与历史 tokens 拼接，经过论文指定的两个 block：

```text
Z <- Z + Expand(SelfAttention(Compact(Z)))
Z <- Z + FFN(Z)
```

`Compact`/`Expand` 使用同一套线性算子处理 query/history 的每个 2×2 patch block；attention 分支缩小 4 倍 token，FFN 保持全分辨率。输出始终是：

```text
[B, 20×30×52, 2048]
```

这些 memory tokens 在 LingBot patch embedding 之后放到 noisy target tokens 前面，等价于沿 latent temporal axis 拼接 20 个 memory frames。主干只投影并返回后面的 21 个 target latent frames。目标 camera 也按 latent frame 编成 action embedding，加到 LingBot 的 timestep modulation input，而不是渲染点云或 RGB pose condition。

## Geometry supervision

训练时从完整历史中均匀抽一个 frame，严格使用论文的 camera query：

```text
ray(u,v) = [world camera origin, normalized world ray direction]
```

ray MLP 加 learnable 2D grid embedding 后，依次经过：

1. 对 memory 的 cross-attention；
2. patch-grid self-attention；
3. FFN 和 VGGT feature projection。

冻结的官方 VGGT-1B 只处理这个被抽中的 frame，取最后一个 aggregator block 的 patch tokens。监督是逐 patch cosine loss：

```text
L = L_flow_matching + 0.05 * L_geometry
```

VGGT 和 geometry head 都不会在推理时运行。

## 启动训练

先安装基础环境，再安装训练期 teacher：

```bash
pip install -r requirements-glibc217-cu118.txt
pip install -r requirements-gim-world.txt
```

修改 `configs/gim_world_roomtour.json` 中的本地 `dataset_root`、`model_dir`、
两个派生 cache 路径和 `output_dir`，然后：

```bash
accelerate launch scripts/train_geometry_aware_memory.py \
  --config configs/gim_world_roomtour.json
```

训练 loop 开始前会打印本地 dataset 路径、条目数、scene reuse 次数、分辨率、
目标长度、MI budget、memory 容量、总参数量、可训练参数量、world size、
effective batch size 和学习率。TensorBoard event 写到 config 指定的
`output_dir`：

```bash
tensorboard --logdir /path/from/config/output_dir
```

主要曲线：

- `train/loss`：总损失；
- `train/flow_loss`：目标视频 flow-matching MSE；
- `train/geometry_loss`：camera-query VGGT cosine loss；
- `train/memory_norm`：memory token 平均 L2 norm；
- `train/history_candidates` / `history_retained`：MI pruning 前后 latent frame 数；
- `train/sigma`：本 step 的 flow noise level。

论文使用 full-backbone、batch size 32、8K steps、learning rate 1e-5。默认 config 保留 `backbone_train_mode=full`、8K steps 和 1e-5；单机显存不足时可以显式设为 `frozen` 做 memory-only ablation，但这不再是论文主实验。

## 启动推理

先修改 `configs/gim_world_local_inference.json`。推理复用与训练完全相同的
causal prefix、scene latent cache、target window、camera normalization、
MI pruning、memory encoder 和 action path：

```bash
python scripts/inference_geometry_aware_memory.py \
  --config configs/gim_world_local_inference.json
```

输出：

- `generated.mp4`：完整 diffusion sampling 得到的 81 帧；
- `ground_truth.mp4`：同一 withheld window；
- `metadata.json`：target indices、候选/保留 memory 数量和具体保留 indices。

`--num_blocks N` 可做真正的连续 rollout。每个 81-RGB block 生成后，它的
21 个 latent 和对应 camera 会 append 到 `DynamicGIMHistory`；下一个 block
开始前重新做 MI pruning 和 `m_t=M(H_t)`。同时输出每个
`generated_block_XXX.mp4` / `ground_truth_block_XXX.mp4` 以及拼接后的总视频。
多 block 模式只允许论文一致的 `prefix` context，防止未来真值混入初始 memory。

推理不会加载 VGGT，也不会执行 geometry head，这与论文一致。

`DynamicGIMHistory` 实现论文公式 `m_t=M(H_t)` 的在线更新语义：每个生成
block 的 latent 和 camera 会 append 到 history，下一次生成前重新执行 MI
pruning 和 memory encoder。它不是把旧 memory slots 递归写回自身，因此不会
把一次压缩误差永久固化；history 增长时，送进 encoder 的数量仍被 `K` 限制。

## 数据 debug

修改 `configs/gim_world_local_debug.json` 后，可在不加载模型的情况下检查一个
挂载 scene 的索引范围、合法 target 和 memory 数量：

```bash
python scripts/debug_geometry_aware_memory_data.py \
  --config configs/gim_world_local_debug.json
```

将 config 中的 `break_after_resolve` 设为 `true`，会在 scene 本地路径解析完成、
尚未索引 RGB/pose/intrinsics 时进入断点。此时没有下载步骤，`local_root`
就是挂载目录中的原始 scene 路径。
