# GIM-World 论文一致性与任务适配审计

审计对象是 *Geometry-Aware Implicit Memory for Video World Models*
（arXiv:2606.02436v1）。论文没有公开可直接移植的官方训练仓库，因此下表区分
论文明确方法、论文未披露工程细节和本项目双轨迹任务适配。

## 论文明确方法

| 论文细节 | 当前实现 |
|---|---|
| 历史先过 VAE 和 diffusion patch embedding | history 共享 LingBot `patch_embedder` |
| camera embedding 只加到 history，memory query 无 pose | `GIMImplicitMemoryEncoder` |
| fixed learnable memory queries | 20 个 latent patch grids |
| memory encoder 两个 blocks | config 固定 depth=2 |
| attention 分支 2×2 Compact/Expand，FFN 全分辨率 | `GIMMemoryEncoderBlock` |
| query/history 使用共享 Compact/Expand | 每 block 一套共享线性算子 |
| query/history rotary grids 分开编号 | 两段 position ids 各从 0 开始 |
| memory 作为 target temporal prefix | LingBot `[memory, noisy query, text]` |
| action embedding 加入 timestep condition | query camera action 进入 `temb_input` |
| geometry query 是 `[origin,direction]` ray | `make_origin_direction_rays` |
| geometry head 为 cross-attn → self-attn → FFN | `CameraQueryableGeometryHead` |
| frozen VGGT patch feature + cosine loss | `VGGTGeometryTeacher` |
| `L = L_FM + 0.05 L_geo` | 默认 `geometry_loss_weight=0.05` |
| pose-time kernel + MI greedy pruning | `MIGreedyPruner` |
| pruning `K=200`，memory 容量 20 frames | 默认 config |
| end-to-end joint training | 默认 `backbone_train_mode=full` |
| VGGT/geometry decoder 推理时丢弃 | inference 不加载 VGGT并删除 geometry head |

GIM attention 使用 PyTorch SDPA；LingBot packed attention 在
FlashAttention3 不存在时也回退到 SDPA。

## 论文未披露的实现选择

| 未披露项 | 当前选择 |
|---|---|
| pose/time kernel 的具体 sigma | position 自动按 scene 尺度；rotation=30°；time=50 latent frames |
| camera vectorization | c2w 3×4 + 归一化 `fx,fy,cx,cy` |
| block norm/activation/bias | pre-RMSNorm、GELU FFN、SDPA |
| VGGT 层 | final aggregator patch tokens，2048D |
| VGGT 预处理 | 官方 crop-mode 的 tensor 等价实现，宽 518、14 对齐 |
| flow sigma distribution | uniform flow sigma，shift 默认 1 |
| 长 capture VAE 工程路径 | Wan 官方 causal state，首帧后每 4 帧一组，跨 CPU read chunk 延续 |

VAE 和 VGGT 不做磁盘缓存是计算策略，不改变模型目标：两者每次从 RGB 在线
运行，结果只存活于当前 scene iteration。

## 明确的双轨迹任务适配

论文任务与本项目数据不完全相同，下列设计不是伪装成论文原超参数：

1. 单个 room-tour 的随机长 window 被拆成两个不重叠的规则采样 phase：一条是
   稀疏 capture，另一条提供 withheld query。它是单轨迹数据对“双轨迹 novel
   view”任务的近似，不宣称是两条真正独立轨迹。
2. window、stride、phase 和 query crop 均随机；pose coverage score 只用于
   诊断，不再挑选最容易的 top-k 样本。query pose 不参与 GIM 的 MI pruning。
3. geometry query 从当前 withheld query block 中抽取，而不是从 memory history
   中抽取。geometry head 只能读 memory 与 camera ray，GT RGB 只供冻结 VGGT
   产生监督，因而直接约束 novel-view 3D 可查询性。
4. 默认一个 scene iteration 只监督一个 49-RGB query chunk，不做没有下游 loss
   的 memory 写回。代码仅在显式设置多个 query blocks 时于 block 之间写回。
5. capture 首帧被设为共同世界原点；MI 时间保留稀疏采样的真实内部间隔
   `0,4s,8s,...`。
6. 原论文用离散游戏 action；room tour 没有动作标签，因此精确 camera pose 和
   intrinsics 被编码为 LingBot action condition。

这套适配的目标是缩小 train/inference gap，而不是更改 GIM memory encoder
本身。

## 验证边界

- Python bytecode compile 覆盖 data、training、inference 和 debug 脚本。
- 单元测试覆盖交错 phase 的均匀 stride/不重叠、window curriculum、MI subset、
  fixed memory shape/gradient、VGGT grid、LingBot memory prefix 和动态 history。
- 真实数值 smoke test 仍需在安装 PyTorch、LingBot checkpoint、VGGT 和一条
  mounted scene 的训练服务器上执行：

```bash
pytest -q tests/test_geometry_aware_memory.py
```

建议第一次运行时用一个 scene list、`num_train_epochs=1`、
`capture_window_max_rgb_frames=321`、`query_blocks=1` 和
`backbone_train_mode=frozen` 验证显存/耗时；这只用于 smoke test，正式实验再
恢复 full backbone 与长 capture curriculum。
