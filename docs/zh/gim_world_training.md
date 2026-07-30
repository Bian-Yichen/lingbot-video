# GIM-World × LingBot-Video：双轨迹长程训练

本分支从 `compat/glibc217-torch26` 建立，模型模块复现
*Geometry-Aware Implicit Memory for Video World Models*（GIM-World），数据任务则
适配为实际使用方式：

1. 输入一条连续、较长的 capture walkthrough；
2. 输入另一条独立 query 轨迹的 camera pose 和 intrinsics；
3. 逐 block 生成 query RGB；
4. 每生成一个 block 就写回 history，重新压缩 memory，再生成下一段。

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

默认从 1000 帧中构造：

- 一个连续 capture 窗口，长度为 `1 + 4k`；
- 一个连续 query 窗口，默认包含 `2 × 81 = 162` 帧；
- capture 与 query 完全不重叠，中间至少隔 32 帧；
- query 的两个 81 帧 block 彼此连续。

query 在原视频时间上可以位于 capture 之前或之后。模型时间始终重新定义为
“先观察完整 capture，再沿 query pose 生成”，因此不会暗中使用原视频中
capture/query 的时间先后关系。

随机窗口不只按时间抽。每个 scene 先生成 128 个合法候选，用 pose 计算 query
中每个相机到 capture 相机集合的最近距离：

```text
cost = normalized_translation_distance
     + rotation_weight * forward_direction_angle
score = median(nearest_cost) + 0.25 * q90(nearest_cost)
```

从 score 最小的 top-8 中随机选一个。这样 query RGB 与 capture RGB 是独立帧，
但 query 位置更可能确实被 capture walkthrough 覆盖，接近“同一房间的另一条
轨迹”，也避免把完全未观测区域当成确定性重建目标。

所有 c2w 都左乘 capture 首帧 c2w 的逆；capture 首帧因此位于原点且旋转为
identity。intrinsics 在 resize + center crop 后同步变换。

## 没有任何派生 feature cache

训练配置中不存在 `latent_cache_root` 或 `teacher_cache_root`：

- 每个 scene iteration 都从 capture RGB 开始，使用一个全新的 Wan causal
  feature state 在线编码整条连续 capture；
- 每个 81-RGB query block 都从空 Wan state 在线编码成 21 个监督 latent；
- 每个 query block 都从该 block 中随机取一张 RGB，在线运行一次冻结的
  VGGT-1B；
- 任何 VAE latent 或 VGGT feature 都不会写盘，也不会在下一 epoch 复用。

`vae_encode_chunk_rgb_frames=81` 只控制 CPU 读取批次。capture 内的 Wan causal
state 不会在这 81 帧边界重置：第一个 RGB 单独编码，之后严格每 4 帧一组，
直到完整 capture 结束。

## 一个 iteration 内的顺序

capture 在线编码后，初始 history 由 capture latent、capture camera 和合成模型
时间 `0,4,8,...` 组成。对每个 query block 依次执行：

1. 对当前 history 做 pose-time GP mutual-information greedy pruning，最多保留
   `K=200` 个 latent frame；query pose 不参与 pruning，所以这不是
   target-conditioned capture retrieval。
2. 将保留的 history latent 经过 LingBot 共享 patch embedding、camera
   embedding 和两层 GIM memory encoder，得到固定 20 latent-grid 的 memory。
3. 当前 81-RGB query block 独立编码为 21 个 target latent，采样 flow sigma 和
   noise，以 query camera action 为条件预测 velocity，计算 flow-matching MSE。
4. 从当前 query block 随机选一个 novel query view。冻结 VGGT 从其 GT RGB
   计算 patch feature；geometry head 只根据 memory 和这张图的 camera rays
   预测 feature，计算逐 patch cosine loss。GT query RGB 不进入 memory。
5. 将当前 block 写回 history，再开始下一个 block。写回值按 schedule 选择：
   - teacher forcing：GT target latent；
   - self-conditioning：由当前单步 flow 预测得到的 detached
     `x0_hat = noisy - sigma * predicted_velocity`。
6. 写回后分配继续递增的模型时间，下一 block 重新 pruning 并重新运行 memory
   encoder。

总损失是所有 query blocks 的平均：

```text
L = mean_blocks(L_flow + geometry_loss_weight * L_geometry)
```

这种设计让同一个 iteration 同时训练初始 memory 使用和动态更新后的 memory
使用，而不是只监督一个和 rollout 无关的孤立 81 帧 clip。

## 默认 schedule

默认 20 epochs，每个 epoch 每个 scene 一次：

| Epoch（从 1 开始） | capture 长度范围 | 预测 latent 写回概率 |
|---:|---:|---:|
| 1 | 257–321 | 0.000 |
| 2 | 333–441 | 0.125 |
| 3 | 421–561 | 0.250 |
| 4 | 513–681 | 0.375 |
| 5 及以后 | 601–801 | 0.500 |

capture 实际长度始终为 `1 + 4k`；默认下界至少是当前上界的 75%，并且不低于
257。前几轮先让
memory/action/geometry 分支学会可解的短 history，再扩展到约 800 RGB（201
latent）和动态预测写回。若训练早期明显不稳定，可先把
`predicted_update_probability_end` 调成 0.25；不建议一开始就 100% 写回预测
latent，因为单个高噪声 flow training point 的 `x0_hat` 远弱于完整 diffusion
rollout。

`gradient_accumulation_steps` 只决定多少个 scene 的梯度合成一次 optimizer
update，不会让 dataset 在一个 epoch 内重复 scene。单 GPU 下严格每 scene
一次；多 GPU dataset 数不能整除 world size 时，Accelerate 最多会填充
`world_size - 1` 个 scene，并在日志中警告。

## 模型形状与注入

默认 480×832 下，capture latent 和 patch token 为：

```text
RGB capture:                 [1, 3, T_rgb, 480, 832]
Wan latent:                  [1, 16, 1+(T_rgb-1)/4, 60, 104]
LingBot patch (1×2×2):       [1, T_latent, 30×52, 2048]
MI retained history:         at most 200 latent frames
GIM fixed memory:            [1, 20×30×52, 2048]
query RGB block:             [1, 3, 81, 480, 832]
query latent block:          [1, 16, 21, 60, 104]
```

camera embedding 只加到 history tokens。GIM memory tokens 在 LingBot
patch embedding 之后作为 temporal prefix 放到 noisy query tokens 前；主干只
返回 query 的 21 个 latent frame。query camera action embedding 加到 timestep
modulation path，不通过点云或 RGB pose rendering 注入。

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

修改 `configs/gim_world_local_inference.json` 后运行：

```bash
python scripts/inference_geometry_aware_memory.py \
  --config configs/gim_world_local_inference.json
```

随机采样时使用与训练相同的 capture/query 轨迹构造器。也可以同时指定
`capture_start`、`query_start` 和可选的 `capture_rgb_frames` 以固定实验。

推理只编码 capture RGB；query 只输入 camera pose/intrinsics。每个生成 block
的 latent 会追加到 `DynamicGIMHistory`，下一 block 前重新执行 MI pruning 和
`m_t=M(H_t)`。query GT RGB 只在生成完成后读取并保存
`ground_truth*.mp4`，绝不参与生成。VGGT 和 geometry head 不在推理时运行。

## 数据 debug

```bash
python scripts/debug_geometry_aware_memory_data.py \
  --config configs/gim_world_local_debug.json
```

输出 capture/query 的内部与源 index 范围、实际 gap、pose coverage score、
latent 数量和 capture 首帧归一化结果。`break_after_resolve` 和
`break_after_sample` 可分别在路径解析后、轨迹采样后进入断点。
