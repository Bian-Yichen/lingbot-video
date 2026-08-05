# GIM-World × LingBot-Video：局部视角检索训练

本分支从 `compat/glibc217-torch26` 建立，模型模块复现
*Geometry-Aware Implicit Memory for Video World Models*（GIM-World），数据任务则
适配为单条 walkthrough 能提供的、尽量接近实际 novel-view 推理的监督：

1. 在 scene 中随机截取一个默认 81 帧的局部 window；
2. 将 window 中间连续 41 帧作为 withheld target trajectory；
3. 从其余 40 帧中用 pose-aware greedy facility location 检索 2–24 张 memory
   views；
4. 每张 RGB 都作为独立单帧通过 Wan VAE，严格得到一帧一个 latent 和一个
   camera pose；
5. 默认一次只监督一个 target chunk，不在该 iteration 内做无效写回。

## 本地数据与时间轴

`dataset_root` 的直接子目录是 scene，数据从挂载目录原地读取，不调用 rclone，
也不复制到 `/tmp`：

```text
<dataset_root>/<item_name>/
├── RGB/
├── chunk_metadata.json
└── vipe/vipe_artifacts/
    ├── pose/video.npz
    └── intrinsics/video.npz
```

RGB、pose 和 intrinsics 只使用源 index `0,5,10,...`。loader 将它们映射成内部
连续 index `0,1,2,...`，所以一个源视频 5000 帧的 scene 通常有 1000 个可训练
frame。下文的帧数均指这个内部时间轴。

## 一个 scene 如何成为一个 iteration

每个 epoch 每个 scene 只取一次。下一 epoch 再访问同一个 scene 时，seed 中包含
epoch，因此会产生另一组轨迹，而不是在一个 epoch 内反复优化同一个 scene。

默认 `local_window_rgb_frames=81`、`target_rgb_frames=41`。target 固定放在局部
window 中间，因此前后各有 20 张候选 view，避免旧版纯向前外插。每个 epoch
重新随机 window，并在 `[2,24]` 内随机 memory view 数量。

检索只使用已有 SLAM camera pose，不读取 target RGB feature。对候选 view `c`
和每个 target view `t` 计算：

```text
cost(c,t) = normalized_position_distance
          + retrieval_rotation_weight * SO(3)_geodesic_angle / pi
sim(c,t)  = exp(-cost(c,t) / retrieval_temperature)
```

随后用 greedy facility location 逐张选择边际 coverage gain 最大的候选。已经被
某张 memory view 覆盖的 target camera 再选相似 view 几乎没有收益，因此少量
memory 会自动分布到 target 轨迹的不同局部，而不是全部挤在一个边界附近。选中
后按时间顺序送入 memory encoder。`retrieval_coverage_score` 越高越好，
`trajectory_overlap_score`（pose cost）越低越好，两者都写入日志。

所有 c2w 都左乘 capture 首帧 c2w 的逆；capture 首帧因此位于原点且旋转为
identity。intrinsics 在 resize + center crop 后同步变换。

## 没有任何派生 feature cache

训练配置中不存在 `latent_cache_root` 或 `teacher_cache_root`：

- 每个 memory RGB 都以 `[B,3,1,H,W]` 的单帧 sample 在线编码；
- 默认 41-RGB target chunk 被拆成 41 个单帧 sample，编码后仍为 41 latent；
- 每个 query block 都从该 block 中随机取一张 RGB，在线运行一次冻结的
  VGGT-1B；
- 任何 VAE latent 或 VGGT feature 都不会写盘，也不会在下一 epoch 复用。

`vae_encode_chunk_rgb_frames=4` 只控制单帧 VAE sample 的 batch size。batch 中
各帧不共享 temporal causal state；推理也逐帧独立 decode，训练和推理完全一致。

## 一个 iteration 内的顺序

memory views 在线编码后，初始 history 由 memory latent、对应 camera 和真实内部
时间偏移组成。默认单个 target chunk 执行：

1. 对当前 history 做 pose-time GP mutual-information greedy pruning，最多保留
   `K=24` 个 latent frame。输入本来只有 2–24 张，因此默认不会再次截断；MI
   pruning 保留用于以后动态 history 扩张。
2. 将保留的 history latent 经过 LingBot 共享 patch embedding、camera
   embedding 和两层 GIM memory encoder，得到固定 20 latent-grid 的 memory。
3. 当前 41-RGB query chunk 编码为 41 个独立 target latent，采样 flow sigma 和
   noise，以 query camera action 为条件预测 velocity，计算 flow-matching MSE。
4. 从当前 query block 随机选一个 novel query view。冻结 VGGT 从其 GT RGB
   计算 patch feature；geometry head 只根据 memory 和这张图的 camera rays
   预测 feature，计算逐 patch cosine loss。GT query RGB 不进入 memory。
5. 计算 `L_flow + 0.05 L_geometry`，反传后结束该 scene iteration。

默认总损失：

```text
L = L_flow + geometry_loss_weight * L_geometry
```

单 block 训练时，当前预测写回 memory 后没有任何后续 loss 会读取它，因此写回
不可能训练动态更新，反而会制造“代码看起来在 self-forcing、实际上梯度目标没
变”的错觉。实现现在只在 `query_blocks>1` 且后面确实还有 block 时写回；默认
`query_blocks=1`、预测写回概率为 0。

这意味着当前 stage 专门学习“少量检索 memory views → 一个连续 withheld target
chunk”。多 chunk rollout 仍由 inference 支持，但如果以后要正式训练生成内容
作为 memory 的鲁棒性，应单独开第二 stage：每个 scene 监督至少两个 query
chunks，第一段完整 diffusion rollout 或高质量预测写回后，第二段 loss 才真正
约束更新后的 memory。本次没有把这个额外困难混入第一阶段。

## 默认 schedule

默认 20 epochs，每个 epoch 每个 scene 一次。所有 epoch 都使用 81 帧局部
window、连续 41 帧 target，并随机取 2–24 张 memory views。当前版本不再使用
长 window curriculum；跨 epoch 的随机性来自 scene 内 window 位置、memory
数量、facility-location tie break、geometry query 和 flow noise。

`gradient_accumulation_steps` 只决定多少个 scene 的梯度合成一次 optimizer
update，不会让 dataset 在一个 epoch 内重复 scene。单 GPU 下严格每 scene
一次；多 GPU dataset 数不能整除 world size 时，Accelerate 最多会填充
`world_size - 1` 个 scene，并在日志中警告。

## 模型形状与注入

默认 480×832 下，capture latent 和 patch token 为：

```text
RGB memory views:            2–24 × [1, 3, 1, 480, 832]
Wan memory latent:           [1, 16, 2–24, 60, 104]
LingBot history patches:     [1, 2–24, 30×52, 2048]
MI retained history:         at most 24 latent frames
GIM fixed memory:            [1, 20×30×52, 2048]
target RGB chunk:            41 × [1, 3, 1, 480, 832]
target latent chunk:         [1, 16, 41, 60, 104]
```

camera embedding 只加到 history tokens。GIM memory tokens 在 LingBot
patch embedding 之后作为 temporal prefix 放到 noisy query tokens 前；主干只
返回 target 的 41 个 latent frame。每个 latent 都有一套一一对应的 camera
pose/intrinsics；camera action embedding 加到 timestep
modulation path，不通过点云或 RGB pose rendering 注入。

## LingBot backbone 的三种训练模式

`backbone_train_mode` 支持三种值：

| 模式 | LingBot backbone | checkpoint 中的 backbone 权重 | 用途 |
|---|---|---|---|
| `full` | 全参数训练 | 完整保存 | 论文原始 joint training 设置，效果基线 |
| `frozen` | 完全冻结 | 不保存 | 仅用于 smoke test / memory-only ablation |
| `lora` | 冻结 base、训练 LoRA | 只保存 LoRA A/B | 当前推荐的低存储训练设置 |

LoRA 默认注入每个 LingBot block 的 attention `to_q/to_k/to_v/to_out`
和 dense FFN `gate_proj/up_proj/down_proj`。对于 1.3B dense backbone、
24 blocks、rank 32，约新增 31.5M 个 backbone 可训练参数。LoRA B 使用零
初始化，因此刚注入时 backbone 数值行为与原 checkpoint 完全一致；随后 flow
loss 和 geometry-memory 路径共同训练 LoRA 与 GIM 新模块。

配置示例：

```json
{
  "backbone_train_mode": "lora",
  "lora_rank": 32,
  "lora_alpha": 32.0,
  "lora_dropout": 0.0,
  "lora_target_modules": [
    "to_q", "to_k", "to_v", "to_out",
    "gate_proj", "up_proj", "down_proj"
  ],
  "save_optimizer_state": true
}
```

`save_optimizer_state=true` 支持完全一致地续训，但 AdamW 状态仍会占用明显
磁盘空间。若 checkpoint 只用于 inference，可以设为 `false`；模型、LoRA 和
训练进度仍会保存，但以后 resume 会使用新初始化的 AdamW 状态。inference 会
从 checkpoint 的 `training_config` 自动恢复 LoRA 结构，不需要在 inference
config 中重复填写 rank 或 target modules。

需要注意：`lora` 是针对 LingBot 嫁接的参数高效适配，不是 GIM-World 论文的
原始全参数训练设置；但它允许生成主干真正学习读取新加入的 memory prefix，
比完全冻结 backbone 更适合正式实验。

## 启动训练

先安装基础环境和官方 VGGT：

```bash
pip install -r requirements-glibc217-cu118.txt
pip install -r requirements-gim-world.txt
```

只需修改 `configs/gim_world_roomtour.json` 中的 `model_dir`、
`dataset_root` 和 `output_dir`：

```bash
accelerate launch scripts/train_geometry_aware_memory.py \
  --config configs/gim_world_roomtour.json
```

开始前会打印 dataset 条目数、每 epoch scene iterations、local retrieval
配置、独立单帧 VAE/VGGT 模式、参数量、memory token 数、world size、effective
scene batch 和学习率。TensorBoard：

```bash
tensorboard --logdir /path/from/config/output_dir
```

主要曲线包括 `loss`、`flow_loss`、`geometry_loss`、`memory_norm`、
`history_candidates`、`history_retained`、`predicted_update_fraction`、
`capture_latent_frames`、`retrieval_coverage_score` 和
`trajectory_overlap_score`。

每个 epoch 默认保存一次 checkpoint。resume 会从 checkpoint 记录的下一个
epoch 开始。

## 推理

`checkpoint` 可以指向 `trainable_components.pt`，也可以指向包含该文件的
checkpoint 目录。配置好固定的 `model_dir`、`dataset_root` 和默认输出目录后，
每次只需要在命令行指定 checkpoint、scene 名称和本次输出目录：

```bash
python scripts/inference_geometry_aware_memory.py \
  --config configs/gim_world_local_inference.json \
  --checkpoint /path/to/checkpoint-epoch-0001-step-00001105 \
  --item_name zx8lnpzDG58_012000_017000.mp4 \
  --output_dir /path/to/inference/zx8_epoch1
```

随机采样时使用与训练相同的 local-window、连续 target 和 pose facility-location
构造器。固定实验必须同时指定 `local_window_start`、`target_start` 和
`memory_view_count`，三者都位于除以 5 后的内部连续时间轴。

`sample_epoch` 仍写入 metadata 用于复现；当前 sampler 不再有 curriculum。
`seed` 同时决定随机 window/memory 检索和初始生成噪声；
保存的 `metadata.json` 记录完整内部 index 和对应的原始 RGB index，可用于复现。

推理只独立编码选中的 memory RGB；target 只输入 camera pose/intrinsics。每个生成 block
的 latent 会追加到 `DynamicGIMHistory`，下一 block 前重新执行 MI pruning 和
`m_t=M(H_t)`。query GT RGB 只在生成完成后读取并保存
`ground_truth*.mp4`，绝不参与生成。VGGT 和 geometry head 不在推理时运行。

默认 `num_blocks=1`，与当前单 query-block 训练完全一致。如果主动设置为大于
1，后续 block 会把前一 block 的生成 latent 写回 history；脚本会明确警告这是
长 rollout 测试，而不是当前 stage 训练时见过的输入。

主要输出：

- `conditioning_capture.mp4`：真正输入 memory 的 2–24 张检索视角；
- `generated.mp4`：去噪得到的 target sequence；
- `ground_truth.mp4`：生成完成后才读取的 withheld target RGB；
- `comparison_gt_generated.mp4`：左侧 GT、右侧生成结果；
- `metadata.json`：采样 index、pose 原点、memory retained 数、checkpoint 和
  scheduler 参数。

## 数据 debug

```bash
python scripts/debug_geometry_aware_memory_data.py \
  --config configs/gim_world_local_debug.json
```

输出 local window、memory/target 的内部与源 index、retrieval coverage、latent
数量和 memory 首帧归一化结果。`break_after_resolve` 和
`break_after_sample` 可分别在路径解析后、轨迹采样后进入断点。
