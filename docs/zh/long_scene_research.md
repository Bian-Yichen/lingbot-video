# 5000 帧场景记忆：研究判断与方案取舍

调研截止 2026-07-23。这里的目标不是一般长视频，而是：从带 SLAM pose 和 DAV3 depth
的长室内 capture 中建立场景状态，再沿多条新轨迹生成，并让结果足够一致以拟合 3DGS。

## 1. 首先排除 capture retrieval

AnyRecon 一类 target-pose → capture-frame retrieval 在本任务中不是合理主干。它把一个
`world state` 问题退化成 top-k evidence selection：

- 5000 帧中存在大量重复、遮挡和跨房间相似外观，top-k 的离散错误会直接丢失事实；
- 每个 target chunk 得到不同证据子集，回环和多轨迹没有同一个持久 scene state；
- 生成历史只能重新加入候选池，不能自然表达“刚生成的内容立即影响下一段”；
- 增大 k 只会重新引入 context length 和 attention dispersion 问题。

因此本分支已经删除 candidate pool、pose retriever、MMR 和 retrieval teacher。完整 capture
按时间顺序流过固定容量 recurrent state，不按 target pose 选择帧。

## 2. 近期长记忆工作提供了什么

| 思路 | 代表工作 | 可借鉴部分 | 仍需解决的问题 |
| --- | --- | --- | --- |
| 层级压缩全部历史 | [Infinite-World](https://arxiv.org/abs/2602.02393) | local→global 递归压缩；固定预算；生成历史自然成为后续记忆 | 单一压缩状态会忘小物体；错误生成可能被永久写入 |
| 历史细节预训练 | [PFP](https://arxiv.org/abs/2512.23851) | 先要求 memory 重建任意过去帧，再用于 autoregressive generation | 需要空间地址，否则只是时间摘要 |
| 压缩 latent/KV 历史 | [RELIC](https://arxiv.org/abs/2512.04040)、[StreamMem](https://arxiv.org/abs/2508.15717) | 固定 token/KV budget；teacher-to-causal/self-forcing | 纯时间位置不足以表达大室内 3D 空间 |
| 多时间尺度 memory | [Composition of Memory Experts](https://arxiv.org/abs/2605.18813)、[Homer](https://arxiv.org/abs/2607.02588) | short/long/spatial state 分工 | 必须规定生成证据如何从短期升级到共享长期 |
| 循环 3D state | [CUT3R](https://arxiv.org/abs/2501.12387)、[LONG3R](https://arxiv.org/abs/2507.18255)、[LingBot-Map](https://arxiv.org/abs/2604.14141) | 每帧更新持久状态；常数内存；多尺度和 gated update | 重建 state 不能直接替代生成模型中的外观记忆 |
| 在线可塑性 | [Titans](https://arxiv.org/abs/2501.00663)、[Gated DeltaNet](https://arxiv.org/abs/2412.06464)、[SlotMem](https://arxiv.org/abs/2607.15772) | learned writer、delta update、surprise/novelty gate | 无节制写入会造成 catastrophic overwrite |
| 保守、可靠性更新 | [MeMix](https://arxiv.org/abs/2603.15330)、[ReCal3R](https://arxiv.org/abs/2607.05356)、[FILT3R](https://arxiv.org/abs/2603.18493)、[PAS3R](https://arxiv.org/abs/2603.21436) | write-less、置信度、运动/新颖性和滤波式更新 | 要把 reliability 与 observed/generated provenance 结合 |
| 超长流式状态 | [HorizonStream](https://arxiv.org/abs/2605.23889)、[LingBot-Map](https://arxiv.org/abs/2604.14141) | 10k+ 帧、常数内存、线性时间 | 本任务还需要输出视频的外观细节和多轨迹一致性 |

最关键的实证来自 Infinite-World：不是“从历史中取几帧”，而是把全部历史递归压入固定
state；同时它的 Revisit-Dense 结果说明，长记忆能力取决于训练中是否真的出现密集回环和
长跨度 revisit，不能只靠增加普通短 clip 数据量。

## 3. 最终方案：Persistent Recurrent Scene Memory

模型维护两个固定大小的状态：

- `slow memory`：共享、持久、target-independent 的场景事实；
- `fast memory`：每条输出轨迹自己的 episodic branch，保存近期生成和尚未确认的新证据。

每个 capture chunk 的 VAE latent、归一化相机 pose 和世界坐标 Plücker rays 变成 observation
tokens。固定 slots cross-attend 到新 observation，再执行 confidence-aware delta update：

```text
proposal = UpdateBlock(memory, observation)
novelty  = RMS(proposal - memory)
gate     = sigmoid(Writer(memory, proposal, novelty, old_conf, evidence_conf))
memory'  = memory + gate * evidence_conf * source_scale * (proposal - memory)
```

capture 全部写入 fast，并每隔若干 chunk consolidate 到 slow。存储量与 500、5000 或
50,000 帧无关；计算量随输入长度线性增长。训练时使用 truncated BPTT，推理时可逐块
`no_grad` 更新。

### 3.1 为什么一定要 fast/slow，而不是一个 recurrent state

一个状态同时承担“立即记住”和“绝不污染”是矛盾的。生成 chunk 必须立刻影响下一 chunk，
但一次 hallucination 又不能修改所有轨迹共享的场景事实。解决办法是 branch-and-commit：

| provenance | fast write | slow consolidation |
| --- | --- | --- |
| `CAPTURE` | 立即 | 正常强度 |
| `GENERATED` | 立即，当前 trajectory branch 可见 | 默认不做；即使做也只有 0.1 倍写入强度 |
| `VERIFIED_GENERATED` | 立即 | 通过多视角/跨轨迹验证后正常写入 |

多条轨迹从同一 base slow state fork，各自维护 fast state。只有当新内容满足跨视角重投影、
depth/visibility 和循环一致性阈值时，才以 `VERIFIED_GENERATED` 合并到共享 slow copy。
这不是禁止生成写回，而是把数据库中的 provisional transaction 与 committed fact 分开。

### 3.2 读 memory 不再做 retrieval

每个 target chunk 始终读取全部 fixed-budget `concat(slow, fast)` tokens。它们只在 LingBot
condition 入口拼接一次，复用原有 joint self-attention；不在每个 DiT block 增加一套
memory attention。target camera pose 通过 per-patch ray bias 控制视角，但不负责选择历史帧。

### 3.3 不做 point-cloud rendering，但保留 3D 教师

推理输入中没有 point-cloud RGB/depth render。SLAM 和 DAV3 用于：

- canonical scene coordinate 和 metric scale；
- ray-addressed observation/target tokens；
- training-only ray-query depth head；
- 遮挡感知的同轨迹/跨轨迹重投影；
- 判定 generated evidence 能否从 provisional 升级为 verified。

这样显式重建误差不会成为 RGB 条件，模型仍被迫内化可从任意 ray 查询的 3D structure。

## 4. 训练必须模拟真实闭环

只用 clean capture 建 memory、再 teacher-force 一个 target clip，会在真正 rollout 时失败。
本实现一次训练 step 做：

1. 完整 capture stream → base slow state；
2. base state → 生成 target chunk 的 `x0`；
3. predicted `x0` 以 `GENERATED` 立即写入 branch fast memory；
4. future rollout chunk 从更新后的 state 生成；
5. clean target 经过同一 writer 得到 teacher state，约束 predicted-write state；
6. 另一条 target trajectory 从同一个 base state 分支，用共同可见表面约束跨轨迹一致性。

另有 PFP 风格的 `memory_reconstruction`：从最终 state 按 capture rays 随机查询历史 VAE
latent，防止 memory 只保留粗略房间语义。

## 5. 必须做的消融

| 消融 | 要回答的问题 |
| --- | --- |
| clean-frame concat | recurrent state 是否真正解决 context scaling |
| AnyRecon/pose top-k retrieval | 固定 world state 是否改善回环和跨轨迹稳定性 |
| 单一 state vs fast/slow | 分层是否同时减少遗忘和错误传播 |
| 禁止 generated write | 在线写回是否改善长 rollout 连续性 |
| generated 直接写 slow | provenance gate 是否阻止 hallucination 扩散 |
| 无 memory reconstruction pretrain | 历史小物体和纹理是否被压缩掉 |
| 无 state-consistency / self-forcing | student rollout 是否出现 memory distribution shift |
| 无 ray address | state 是否退化成 pose-free 外观摘要 |
| 无 depth/occlusion teacher | 门洞、遮挡边界和回环是否恶化 |
| 单轨迹训练 | paired training 对 3DGS liftability 的贡献 |
| 128/256/512/1024 slow slots | 质量、吞吐和遗忘的 Pareto 曲线 |
| 4/8/16 chunk 与 4/8/16 consolidation interval | update granularity 的影响 |

模型选择不能只看 FVD。至少报告随 100/500/1000/5000 帧的历史重建、回环重投影、跨轨迹
一致性，以及用指定 poses 拟合 3DGS 后的 withheld-view 指标。
