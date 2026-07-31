# GIM-World × LingBot-Video：交错稀疏视角长程训练

本分支从 `compat/glibc217-torch26` 建立，模型模块复现
*Geometry-Aware Implicit Memory for Video World Models*（GIM-World），数据任务则
适配为单条 walkthrough 能提供的、尽量接近实际 novel-view 推理的监督：

1. 先在 scene 的 1000 帧内部时间轴上随机截取一个长 window；
2. 在 window 内每隔 2 或 3 帧取一个固定相位，组成稀疏 capture video；
3. 从另一个固定相位中截取一段 query，只输入 camera pose 和 intrinsics；
4. 默认一次只监督一个固定长度 query chunk，不在该 iteration 内做无效写回。

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

默认算法如下。设随机 stride 为 `s∈{2,3}`，capture RGB 数量为
`T_cap=1+4k`：

```text
window span = 1 + (T_cap - 1) * s
capture     = window_start + [0, s, 2s, ...]
query pool  = window_start + phase + [0, s, 2s, ...]
phase       ∈ {1, ..., s-1}
```

query pool 与 capture 使用同一个 stride、不同 phase，因此两者都是时间间隔
一致的视频流，并且 RGB 帧严格不重叠。stride=2 时正好是偶/奇帧二分：例如约
300 帧的 window 会产生约 150 帧 capture 和约 150 帧 withheld pool。训练从
withheld pool 随机截取固定 49 RGB 作为 query；49 是最接近 50 且满足 Wan VAE
`1+4k` 约束的长度，编码后恰好是 13 latent frames。

stride=3 时没有把所有余帧按 `1,2,1,2...` 的不规则间隔伪装成一个视频，而是
随机选择 phase=1 或 phase=2，得到一条规则下采样的 query 轨迹。未选中的另一
phase 本 iteration 不使用。

旧版的 128 组候选、pose top-8、capture/query 前后隔离逻辑已经删除。window
start、stride、query phase 和 query crop 都直接随机；pose coverage score 仅作
TensorBoard 诊断，不参与数据筛选。这样不会长期偏向相机几乎静止、最容易重建
的区间。

所有 c2w 都左乘 capture 首帧 c2w 的逆；capture 首帧因此位于原点且旋转为
identity。intrinsics 在 resize + center crop 后同步变换。

## 没有任何派生 feature cache

训练配置中不存在 `latent_cache_root` 或 `teacher_cache_root`：

- 每个 scene iteration 都从 capture RGB 开始，使用一个全新的 Wan causal
  feature state 在线编码整条规则下采样的 capture；
- 默认 49-RGB query chunk 从空 Wan state 在线编码成 13 个监督 latent；
- 每个 query block 都从该 block 中随机取一张 RGB，在线运行一次冻结的
  VGGT-1B；
- 任何 VAE latent 或 VGGT feature 都不会写盘，也不会在下一 epoch 复用。

`vae_encode_chunk_rgb_frames=81` 只控制 CPU 读取批次。capture 内的 Wan causal
state 不会在这 81 帧边界重置：第一个 RGB 单独编码，之后严格每 4 帧一组，
直到完整 capture 结束。

## 一个 iteration 内的顺序

capture 在线编码后，初始 history 由 capture latent、capture camera 和其真实
内部时间偏移组成。若 stride=`s`，latent 的时间为 `0,4s,8s,...`，而不是把稀疏
capture 错当成原始帧率。默认单个 query chunk 执行：

1. 对当前 history 做 pose-time GP mutual-information greedy pruning，最多保留
   `K=200` 个 latent frame；query pose 不参与 pruning，所以这不是
   target-conditioned capture retrieval。
2. 将保留的 history latent 经过 LingBot 共享 patch embedding、camera
   embedding 和两层 GIM memory encoder，得到固定 20 latent-grid 的 memory。
3. 当前 49-RGB query chunk 独立编码为 13 个 target latent，采样 flow sigma 和
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

这意味着当前 stage 专门学习“长稀疏 capture memory → 一个 withheld query
chunk”。多 chunk rollout 仍由 inference 支持，但如果以后要正式训练生成内容
作为 memory 的鲁棒性，应单独开第二 stage：每个 scene 监督至少两个 query
chunks，第一段完整 diffusion rollout 或高质量预测写回后，第二段 loss 才真正
约束更新后的 memory。本次没有把这个额外困难混入第一阶段。

## 默认 schedule

默认 20 epochs，每个 epoch 每个 scene 一次：

| Epoch（从 1 开始） | capture window 范围 | stride | query | 预测写回 |
|---:|---:|---:|---:|---:|
| 1 | 257–321 | 随机 2–3 | 49 RGB / 13 latent | 0 |
| 2 | 367–490 | 随机 2–3 | 固定 | 0 |
| 3 | 495–660 | 随机 2–3 | 固定 | 0 |
| 4 | 622–830 | 随机 2–3 | 固定 | 0 |
| 5 及以后 | 750–1000 | 随机 2–3 | 固定 | 0 |

表中是目标 window 边界；实际 span 会选择满足
`span=1+(T_cap-1)s`、`T_cap=1+4k` 的最近合法值。最大 window 下，stride=2
约产生 497 个 capture RGB / 125 个 latent，stride=3 约产生 333 个 capture RGB
/ 84 个 latent。默认 MI budget `K=200` 因而通常不会在输入端截断这些历史；
主要压缩由 20-grid implicit memory encoder 完成。这正好用于先验证 memory
是否能从越来越大的 observation set 中找到 query 所需局部信息。

`gradient_accumulation_steps` 只决定多少个 scene 的梯度合成一次 optimizer
update，不会让 dataset 在一个 epoch 内重复 scene。单 GPU 下严格每 scene
一次；多 GPU dataset 数不能整除 world size 时，Accelerate 最多会填充
`world_size - 1` 个 scene，并在日志中警告。

## 模型形状与注入

默认 480×832 下，capture latent 和 patch token 为：

```text
RGB capture:                 [1, 3, T_cap, 480, 832]
Wan latent:                  [1, 16, 1+(T_cap-1)/4, 60, 104]
LingBot patch (1×2×2):       [1, T_latent, 30×52, 2048]
MI retained history:         at most 200 latent frames
GIM fixed memory:            [1, 20×30×52, 2048]
query RGB chunk:             [1, 3, 49, 480, 832]
query latent chunk:          [1, 16, 13, 60, 104]
```

camera embedding 只加到 history tokens。GIM memory tokens 在 LingBot
patch embedding 之后作为 temporal prefix 放到 noisy query tokens 前；主干只
返回 query 的 13 个 latent frame。query camera action embedding 加到 timestep
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

开始前会打印 dataset 条目数、每 epoch scene iterations、capture/query
schedule、在线 VAE/VGGT 模式、参数量、memory token 数、world size、effective
scene batch 和学习率。TensorBoard：

```bash
tensorboard --logdir /path/from/config/output_dir
```

主要曲线包括 `loss`、`flow_loss`、`geometry_loss`、`memory_norm`、
`history_candidates`、`history_retained`、`predicted_update_fraction`、
`capture_latent_frames` 和 `trajectory_overlap_score`。

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

随机采样时使用与训练相同的 window/stride/phase/query-crop 构造器。固定实验
必须同时指定 `capture_start`、`query_start`、`capture_rgb_frames` 和
`capture_frame_stride`，这四个参数都位于除以 5 后的内部连续时间轴。

若没有显式指定 `sample_epoch`，脚本读取 checkpoint 的 `next_epoch`，使用该
checkpoint 最近完成 epoch 的 capture-window curriculum。例如 epoch-1
checkpoint 默认仍采样 257–321 帧 window，而不会错误地跳到最终的
750–1000 帧阶段。`seed` 同时决定随机 capture/query 构造和初始生成噪声；
保存的 `metadata.json` 记录完整内部 index 和对应的原始 RGB index，可用于复现。

推理只编码 capture RGB；query 只输入 camera pose/intrinsics。每个生成 block
的 latent 会追加到 `DynamicGIMHistory`，下一 block 前重新执行 MI pruning 和
`m_t=M(H_t)`。query GT RGB 只在生成完成后读取并保存
`ground_truth*.mp4`，绝不参与生成。VGGT 和 geometry head 不在推理时运行。

默认 `num_blocks=1`，与当前单 query-block 训练完全一致。如果主动设置为大于
1，后续 block 会把前一 block 的生成 latent 写回 history；脚本会明确警告这是
长 rollout 测试，而不是当前 stage 训练时见过的输入。

主要输出：

- `conditioning_capture.mp4`：真正输入 memory 的稀疏 capture sequence；
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

输出 window/capture/query 的内部与源 index 范围、stride/phase、pose coverage
score、latent 数量和 capture 首帧归一化结果。`break_after_resolve` 和
`break_after_sample` 可分别在路径解析后、轨迹采样后进入断点。
