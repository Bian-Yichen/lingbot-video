# GIM-World 论文一致性审计

审计对象是 arXiv:2606.02436v1（2026-06-01）。该 PDF 共 12 页，正文到第 9 页，之后是参考文献；当前版本没有 appendix。项目页 `https://gim-world.github.io/` 在实现时没有提供官方代码仓库，因此不存在可以直接移植的作者实现。

## 论文明确给出并原样实现

| 论文细节 | 实现 |
|---|---|
| 历史先过 VAE 和 diffusion patch embedding | `GIMWorldLingBotModel.patchify_history` 共享 LingBot `patch_embedder` |
| camera embedding 只加到 history，memory queries pose-free | `GIMImplicitMemoryEncoder.forward` |
| fixed learnable memory queries | query 数严格等于 20 个 latent patch grids |
| memory encoder 只有 2 个 blocks | config 拒绝 `depth != 2` |
| attention-only 2×2 compact，FFN full resolution | `GIMMemoryEncoderBlock` |
| Compact/Expand 对 query/history 共享 | 每个 block 只有一对共享 projection |
| query/history 使用分离 rotary grids | 两段 position ids 分别从 0 开始 |
| memory 沿 temporal axis 与 target 拼接 | LingBot forward 的 `[memory, noisy target, text]` |
| target action embedding 加 timestep embedding | `video_action_embeds` 进入 `temb_input` |
| geometry query 是 `[origin, direction]` 6D ray | `make_origin_direction_rays`，不是 Plücker ray |
| geometry head 顺序是 cross-attn → self-attn → FFN | `CameraQueryableGeometryHead` |
| 每 step 均匀抽一个 historical view | `RoomTourSample.geometry_query_index` |
| VGGT frozen、逐 patch cosine loss | `VGGTGeometryTeacher` + `gim_training_step` |
| `L = L_FM + 0.05 L_geo` | 默认 `geometry_loss_weight=0.05` |
| pose-time RBF kernel 和 MI greedy log variance ratio | `pose_time_kernel`、`MIGreedyPruner` |
| pruning budget `K=200` | 默认 config |
| memory 容量等于 20 latent frames | 默认 config |
| 8K steps、1e-5、end-to-end joint training | 默认 config |
| teacher/head inference discard | inference 只加载 memory/action/backbone 所需权重 |

所有 GIM attention 都可通过 PyTorch SDPA 执行。LingBot packed attention 在 FlashAttention3 缺失时也会逐 sequence 回退到 SDPA。

## 论文没有披露、必须显式确定的部分

这些不是从作者代码复制的值，配置中均可见：

| 未披露项 | 当前决定 | 原因 |
|---|---|---|
| `sigma_p, sigma_r, sigma_t` 数值 | position 用 scene median；rotation=30°；time=50 latent frames | 论文只给公式 |
| `E_c(c_i)` 的 camera vectorization | c2w 3×4 + 按训练分辨率归一化的 4 个 intrinsics | 支持数据中的变化内参 |
| memory block 的 norm、激活和 bias | pre-RMSNorm、GELU FFN、SDPA | 论文只给残差级公式 |
| VGGT 的具体层 | final aggregator patch tokens，2048D | 论文只写 “VGGT encoder feature map” |
| VGGT 图像预处理 | 官方 crop-mode tensor 等价实现，宽 518、尺寸对齐 14 | 遵循 VGGT 官方仓库 |
| 长视频 VAE 如何避免 OOM | 官方 Wan `_encode` 调度：首帧 + 连续 4 帧组，跨整条 scene 保留每层 causal feature cache；81 帧只作 CPU 预读 | 论文未说明工程缓存 |
| target VAE 边界 | 每个监督 81-RGB clip 从空 causal state 独立编码并缓存，不从整场 latent 硬切 | 与推理时独立解码的首 latent 语义一致 |
| flow timestep distribution | uniform flow sigma；shift 默认 1 | 论文只给 flow objective |

## 数据任务适配，不伪装成论文原设定

论文在 MIND 上使用 action-conditioned causal rollout，历史是 target 之前的
observation。因此默认 `context_policy=prefix`。这样训练 target 不可能通过
未来的 causal VAE latent 泄漏到 memory。离线完整-capture 实验可显式选择
`all_except_target`，但代码强制排除 target 前后至少 128 个 RGB frame；这
是针对 room-tour 数据的防泄漏适配，不是论文超参数。

论文使用离散游戏 action；当前数据没有 WASD action，只有精确 camera pose 和 intrinsics。这里把目标 camera encoding 当作 action embedding加入 timestep path。这是必要的 LingBot/room-tour 适配，memory 模块、注入位置和 geometry supervision 不变。

论文报告 batch size 32。1000-RGB room tour、LingBot 1.3B、20-frame memory tokens 和 VGGT-1B 的单卡 batch 32 不现实；代码支持 Accelerate 多卡和 gradient accumulation，启动日志会打印 effective batch size。默认保持 full backbone；`frozen` 仅作为显存受限 ablation。

## 当前验证边界

- Python AST/bytecode compile 覆盖全部新增脚本和模块。
- 单元测试覆盖 MI subset、fixed memory shape/gradient、VGGT grid，以及 memory temporal-prefix 注入后只返回 target shape。
- 当前开发容器没有 PyTorch，因此这里不能执行 CUDA 数值 smoke test。必须在项目的 `lingbot-video` 环境运行：

```bash
pytest -q tests/test_geometry_aware_memory.py
```

训练前建议先将 `max_train_steps` 设为 2、`backbone_train_mode=frozen` 做数据/shape smoke test，再恢复论文配置。
