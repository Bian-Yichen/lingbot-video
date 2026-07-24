# LingBot-Video Latent Spatial Memory

本分支从 `compat/glibc217-torch26` 独立开发，将论文 *Latent Spatial Memory
for Video World Models*（Mirage）适配到 LingBot-Video，并直接读取
`h:bianyichen/AnyReconProDataset_labeled/` 中的 VIPE 标注。

## 1. 方法与论文的对应关系

论文中的 memory 是一组带 VAE latent feature 的世界坐标点：

\[
\mathcal M=\{(\mathbf p_i,\mathbf f_i)\},\qquad
\mathbf p_i\in\mathbb R^3,\quad \mathbf f_i\in\mathbb R^C.
\]

本实现保留下列全部核心机制：

1. 使用 metric depth 将每个 latent cell 反投影到世界坐标，并将原生 VAE
   latent token 写入该点。
2. 给定目标 `c2w + K`，在 latent 网格投影所有 memory points，用 z-buffer
   选择每个 cell 最前方的 feature。
3. 未投影到点的位置填零，并额外输出 visibility mask，区分“未观察区域”和
   “数值恰好为零的已观察 feature”。
4. 使用由主干对应 block 初始化的 ControlNet-style side branch，并通过
   zero-initialized output projection 在八个深度位置注入 LingBot DiT。
   侧支路将当前 noisy target 与同视角 memory token 分别经过共享 patch
   embedding 后相加，因此每个 denoising step 都依赖当前状态，同时没有额外
   bridging encoder。
5. 将 noisy target、clean preceding overlap、clean reference 分配到不同的
   temporal RoPE 区段，实现 segment-aware rotary encoding。
6. 第一阶段冻结主干和 VAE，只训练 side branch；第二阶段加入 self-attention
   `q/k/v/o` rank-64 LoRA，与 side branch 联合训练。
7. 按 chunk 自回归生成；每个 chunk 完成后将新 latent 写回 3D memory，下一
   chunk 立即读取更新后的 memory。
8. depth 采用论文附录验证最优的 bilinear downsampling；缓存写入前剔除无效
   深度和深度不连续边界。

## 2. 针对 LingBot 与本数据的必要适配

论文 backbone 是 Wan2.2-TI2V-5B，本分支使用 LingBot-Video：

| 项目 | Mirage 论文 | 本实现 |
|---|---:|---:|
| VAE spatial stride | 16 | 8 |
| latent channels | 48 | 16 |
| DiT blocks | 30 | 24 |
| side branch 注入 | 0,4,...,28 | 0,3,...,21 |
| 每 chunk latent 帧 | 9 | 9 |
| 每 chunk RGB 帧 | 33 | 33 |

LingBot 原始发布模型没有 camera-control branch，因此将目标相机的 Plücker rays
`[direction, origin × direction]` 投影为 side-branch token bias。Memory readout
提供已观察区域的外观与几何，Plücker rays 负责未观察区域的相机控制。

论文推理时使用 DA3 为生成帧重新估深度，并用 Qwen+SAM 排除动态物体。本数据是
静态室内 room tour，训练时已有 VIPE metric depth，因此：

- 训练不调用 DA3、Qwen 或 SAM；
- 用 VIPE 有效深度范围和相对深度边界过滤代替动态区域过滤；
- 对已有 memory 可见区域执行多视角深度一致性门控：新深度相对误差默认超过
  15% 时拒绝写入；未观察区域仍可写入，保证场景可以继续扩展；
- 新增 `LatentMetricDepthHead`，用 VIPE depth 监督；
- 推理更新 memory 时直接使用最终生成 latent 和预测 metric depth。

这不仅移除了全部外部模型，也省掉了论文 update 阶段的 decode-reencode。

## 3. 每个 iteration 如何从 5000 帧 item 采样

不会把 5000 帧同时送入网络，也不会每个 iteration 重复下载 item。

默认一个 item 的处理方式：

1. rclone 一次性下载该 item 中训练需要的最小子集：`RGB/`、depth ZIP/shards、
   pose NPZ、intrinsics NPZ 和 metadata。
2. 在节点本地 cache 中保留该 item，默认连续产生 32 个 iteration。
3. 每次从目标帧之前最长 4096 帧的历史中分层抽取 48/64 个 capture keyframes。
   分层抽样覆盖整条历史，不只保留最近帧。
4. 目标取连续 65 张 RGB。LingBot VAE temporal stride 为 4，因此得到 17 个
   latent frames，拆成两个 `9-latent` chunk，相邻 chunk 重叠一个 clean latent。
5. capture 的最后一帧与目标第一帧相同，作为第一个 chunk 的 clean preceding
   overlap；capture 第一帧作为 clean reference。
6. 第一个 chunk 预测后立刻更新 memory；第二个 chunk 使用已经更新的 memory。

因此，“长程”来自跨最多数千原始帧构建的 persistent 3D latent cache，而目标视频
仍保持连续帧，符合预训练视频模型的局部运动分布。随着同一 item 被多次采样，
不同 iteration 会覆盖不同历史范围与目标位置。

## 4. 输入与监督

每个 batch 的输入为：

- `capture_rgb`: `[B,N,3,H,W]`
- `capture_depth`: `[B,N,H,W]`
- `capture_c2w`: `[B,N,4,4]`
- `capture_intrinsics`: `[B,N,3,3]`
- `target_rgb`: `[B,3,65,H,W]`
- target latent 时刻对应的 depth/c2w/intrinsics，共 17 帧

监督包括：

1. 两个 chunk 的 target-frame flow-matching loss；每个 chunk 第一 latent 是
   clean overlap，不计入 flow loss。
2. 生成 clean latent 上的 metric log-depth loss：scale-invariant 项加 metric
   Smooth-L1 项。
3. `readout_error` 仅作为数据/几何监控指标，比较可见 cell 的 memory readout
   与目标 clean latent，不参与优化。

Stage 1 使用 ground-truth latent/depth 更新下一 chunk 的 memory；Stage 2 默认
以 0.5 概率使用模型预测更新，训练生成误差进入后续 memory 后的鲁棒性。

## 5. Pose 与分辨率

VIPE 文件约定：

- `pose/video.npz`: `data` 为 OpenCV camera-to-world，`inds` 为帧号；
- `intrinsics/video.npz`: `data` 为 `[fx,fy,cx,cy]`，对应原始 RGB 分辨率；
- `depth/video.zip` 或 `depth/video.zip.parts/*.zip`: EXR `Z` channel；
- `.zip.partial` 自动忽略，已完成 shard 仍可用于训练。

每个 sample 都执行：

\[
T_i' = T_0^{-1}T_i,
\]

其中 \(T_0\) 是该 sample 的第一个 capture `c2w`。所以首 capture 相机的位置为
原点、旋转为单位阵，capture 与 target 使用同一归一化世界坐标。

不同原始分辨率统一采用 aspect-preserving resize + center crop。内参同步变换：

\[
f_x'=s_xf_x,\quad f_y'=s_yf_y,\quad
c_x'=s_x(c_x+0.5)-0.5-l,\quad
c_y'=s_y(c_y+0.5)-0.5-t,
\]

其中 \(l,t\) 是 crop 左上角；这里显式保留了 `align_corners=False`/PIL resize
的 half-pixel convention。Depth 使用相同视场变换，metric z 值不缩放。

## 6. 安装与训练

```bash
pip install -r requirements-training.txt
```

确保 `rclone config file` 能找到 `h:`，然后：

```bash
python scripts/inspect_latent_spatial_memory_dataset.py \
  --item_name _750e401Wkl_004000_009000.mp4

MODEL_DIR=/path/to/LingBot-Video-Dense \
CONFIG=configs/latent_spatial_memory_stage1.json \
NUM_PROCESSES=8 \
bash scripts/train_latent_spatial_memory.sh
```

Stage 1 完成后运行 Stage 2：

```bash
MODEL_DIR=/path/to/LingBot-Video-Dense \
CONFIG=configs/latent_spatial_memory_stage2.json \
NUM_PROCESSES=8 \
EXTRA_ARGS="--init_component_checkpoint \
outputs/latent_spatial_memory_stage1/final/trainable_components.pt" \
bash scripts/train_latent_spatial_memory.sh
```

默认只保存 `controlnet + depth head + LoRA`，不会在每个 checkpoint 复制完整
1.3B backbone。若需要 Accelerator/FSDP 的精确 optimizer resume，增加
`--save_accelerator_state`，并用
`--resume_from_checkpoint checkpoint/accelerator_state` 恢复。

## 7. 验证推理

下面从同一 VIPE item 的长 capture history 出发，沿标注轨迹生成 257 帧；推理只
读取 camera pose/intrinsics，不读取目标 RGB/depth：

```bash
python scripts/inference_latent_spatial_memory.py \
  --model_dir /path/to/LingBot-Video-Dense \
  --checkpoint outputs/latent_spatial_memory_stage2/final/trainable_components.pt \
  --dataset_root h:bianyichen/AnyReconProDataset_labeled/ \
  --item_name _750e401Wkl_004000_009000.mp4 \
  --target_start 4096 \
  --num_frames 257 \
  --output outputs/mirage_validation.mp4
```

同时输出 `mirage_validation.depth.npz`，包含生成更新使用的预测 depth、归一化
camera poses 与 intrinsics。

## 8. 关键配置建议

- 先用 `480×832, capture_frames=48, rollout_chunks=2` 验证训练正确性。
- Stage 2 再将 capture 增至 64；不建议一开始将 5000 帧全部编码。
- `memory_voxel_size=0` 最忠实于论文；长推理可设置约 `0.02 m`，每个 voxel
  保留最新且最高置信 observation，不平均 latent feature。
- 如果单个 item 很大，继续提高 `samples_per_item`，以摊薄下载成本。
- item 中完整 depth 不足以覆盖 `history_min_frames` 时会跳过，不会误用
  `.partial` 文件。
