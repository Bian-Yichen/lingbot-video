# MIRAGE 论文一致性审计

本文档区分三类内容：论文/附录中明确规定的机制、从 Wan2.2 到 LingBot 必须做的
等价映射，以及本数据与“无外部模型”要求带来的改动。对照材料包括论文正文、
Appendix C、Algorithm 1，以及作者公开的
[LatentSpatialMemory](https://github.com/microsoft/LatentSpatialMemory) 实现。

## 论文规定并原样保留

| 论文规定 | 本实现 |
|---|---|
| 原生 VAE latent-attributed 3D points | `LatentSpatialMemory` 存储 `(world_xyz, C-channel latent)` |
| target pose 上 latent-resolution z-buffer readout | `LatentSpatialMemory.read` |
| zero fill + visibility mask | readout 同时返回 `features/visibility/depth` |
| VACE/ControlNet recurrent side branch | `before_projection(memory)+backbone_tokens`，side blocks 递归运行 |
| zero-init hint | 每个 `after_projections[i]` 权重和 bias 为零 |
| 在指定主干层之后注入 | `block_control_residuals` 在对应 block forward 后相加 |
| 单次 `[reference,target,preceding]` forward | `LingBotVideoLatentMemoryModel.forward` 按该顺序拼接 |
| clean context timestep = 0 | reference/preceding 以及 target overlap 的 timestep 为零 |
| segment-aware RoPE | target、preceding、reference 使用三个不相交 temporal 区段 |
| 9 latent / 33 RGB chunk | dataset 每次只监督一个 33-frame target window |
| one-latent overlap | target latent 0 来自前一个 causal capture clip 的最后 latent |
| flow loss 仅作用于 target | wrapper 只返回 target slice，loss 排除 clean overlap |
| 两阶段训练 | Stage 1 side branch；Stage 2 side branch + self-attention LoRA；用户要求的 depth head 两阶段均训练 |
| LoRA `{q,k,v,o}`, rank/alpha 64, dropout 0.05 | PEFT 配置一致 |
| AdamW、`beta=(0,.999)`、`wd=1e-3`、cosine、bf16、GC、text dropout .2 | 训练脚本和 JSON 配置一致 |
| 40-step inference、CFG off | 推理脚本默认 40 steps，单 conditional pass |

## LingBot 等价映射

Wan2.2-TI2V-5B 有 30 个 block，论文在
`{0,4,8,12,16,20,24,28}` 注入。LingBot 有 24 个 block，因此保持八个等间距
深度位置，映射为 `{0,3,6,9,12,15,18,21}`。Side block 均从对应 LingBot block
深拷贝初始化。

Wan 的 text 是 cross-attention context；LingBot 将 video/text token 放入同一个
self-attention。为保留 clean/noisy frame-wise timestep，Transformer 现在接受
`[B,T]` timestep：video token 使用所属 frame 的 timestep，text token 使用该
sample 最大 target timestep。标量 `[B]` timestep 路径仍与原 LingBot 推理兼容。

论文 Appendix C 同时写了“readout 与 mask concat”和“48-channel、共享主干 patch
embedding”。直接把 1-channel mask 拼到 48-channel latent 会与后一句冲突。这里
让 48-channel latent 经过共享 patch embedding，并让 visibility mask 经过独立的
zero-init linear 后相加；因此 latent 路径没有 bridging encoder，同时未知零值仍可
与真实零 latent 区分。作者仓库当前的 96-channel dummy-score VACE 路径与论文
Appendix C 的 48-channel 描述不同，本分支以论文为准。

## 用户数据要求带来的有意改动

- 基础模型从 Wan2.2 换成 LingBot-Video。
- 不调用 DA3/MapAnything、Qwen 或 SAM。训练 depth 来自 VIPE；推理 memory
  update 使用 `LatentMetricDepthHead`。
- 数据没有可靠动态/sky mask，所以使用 depth validity、depth edge 和已有 memory
  的多视角深度一致性门控。若后续加入 dynamic mask，可直接并入 `valid_mask`。
- Camera pose 除了用于 3D memory readout，还以 zero-init Plücker adapter 提供未观察
  区域的相机控制。这是 LingBot 没有原生 camera-control branch 时的必要补充。
- 本地挂载数据没有预计算 LMDB latent，因此训练时由冻结 VAE 编码。每个 worker
  直接读取 scene，并让一个 item 连续复用多个 iteration；RGB 以 uint8 预加载后
  再在 GPU 上归一化。后续可把 clip latent 做成本地持久 cache，而不改变模型输入
  或监督。

这些改动均独立于 VACE 注入拓扑和论文 flow objective，checkpoint 中也分别以
`controlnet`、`depth_head` 和 LoRA 参数保存。
