# Active World Memory

Active World Memory treats visual recall for novel-view video generation as a
sequential information-gathering problem.  It is built around the unchanged
LingBot-Video DiT from `compat/glibc217-torch26`; it does not render a point
cloud and does not require VGGT, DAV, DINO, or another external teacher.

## What is different from one-shot retrieval

For every target chunk, the target camera trajectory first initializes a
spatio-temporal Evidence Canvas.  The memory agent then repeatedly performs:

1. choose a learned interrogation type (`structure`, `identity`, `appearance`,
   `occlusion`, `context`, `loop_closure`, `expand_temporal`, or
   `verify_complement`);
2. choose the most useful uncertain target region;
3. use that region token to search a coarse temporal episode, then match it
   against the candidate views' low-resolution patch banks;
4. inspect the selected view's patch tokens and update the Canvas;
5. either retrieve again or emit `STOP`.

The next query is conditioned on the updated Canvas, so this is not a renamed
Top-K selector.  `expand_temporal` biases the next action toward neighboring
episodes, while `verify_complement` penalizes near-duplicate evidence.  A
gated Canvas update can reject evidence that does not improve the current
belief.

## No target leakage

At inference, the policy sees only target camera poses/intrinsics, memory RGB,
the current Canvas, selected evidence, and uncertainty.  Target RGB/latent is
not accepted by the policy API.  During training, target latent is used only:

- as the target of the World Critic;
- to label counterfactual oracle actions by information gain;
- as the flow-matching target of LingBot-Video.

The information-gain reward is

`L_critic(E_k-1) - L_critic(E_k) - retrieval_cost`.

This lets the agent learn when another memory access is worth its compute.

## Hierarchical memory and long captures

Every capture view receives a cheap low-resolution patch/view key.  Consecutive
views are pooled into episode keys.  Only the 2--24 views selected by the agent
are encoded independently by the full Wan VAE.  Independent one-frame VAE
encoding preserves an exact camera pose per evidence latent and avoids temporal
mixing between unrelated retrieved views.

Training samples a stratified 48--128-view candidate pool from a progressively
longer capture span.  Across epochs the pool changes.  Inference defaults to
`--full_capture_memory`: every non-target capture thumbnail can be indexed, but
full-resolution RGB is decoded only for selected views.  This keeps a 5000-view
capture tractable without pretending that all 5000 high-resolution VAE maps fit
in the DiT context.

## LingBot condition injection

Selected evidence, the final Evidence Canvas, and target camera tokens are
resampled to a fixed token budget.  They are projected to LingBot's text
conditioning width and appended to prompt tokens.  LingBot's native joint
self-attention therefore consumes the condition at every transformer block.
No global point cloud or rendered proxy is supplied.

The default configs keep LingBot's original weights frozen and train native
rank-16 LoRA updates on every block's attention Q/K/V/O projections.  This is
stronger than soft-token-only adaptation but keeps checkpoints small and has no
PEFT dependency.  `backbone_train_mode` can also be set to `frozen` or `full`;
LoRA tensors are saved separately from the new memory components.
The configs set `fused_qkv_linear=false`, because LingBot's optional fused QKV
shortcut reads base weights directly and would bypass Q/K/V wrapper forwards.

## Three training stages

### Stage 1: retriever and critic bootstrap

A camera-frustum proximity score supplies a soft retrieval teacher.  It is used
only as a bootstrap label, never as generator input.  The critic learns to
predict pooled target VAE features from geometrically diverse evidence.

```bash
accelerate launch scripts/train_active_world_memory.py \
  --config configs/active_world_memory_stage1_retriever.json
```

### Stage 2: counterfactual oracle imitation

For each state, every learned query type proposes candidates for the currently
most erroneous Canvas region.  Their union is independently added to the
Canvas.  The candidate that most reduces target-feature prediction error is the
oracle action; the resulting query-type, region, episode, view, and `STOP`
decisions are imitated as a complete hierarchical action.  Non-positive gain
teaches `STOP` after the minimum evidence budget.

```bash
accelerate launch scripts/train_active_world_memory.py \
  --config configs/active_world_memory_stage2_imitation.json \
  --resume /path/to/stage1/checkpoint-iter-XXXXXXXX
```

### Stage 3: group-relative policy optimization

Several retrieval rollouts share one memory index.  Their group-normalized
information-gain returns train the discrete policy, with an entropy term and
explicit retrieval cost.  The first on-policy evidence sample conditions one
flow-matching generator pass; GT reward is never used to choose the generator
condition.  This avoids four complete DiT passes per iteration.
Two target blocks enable generated/GT latent views from block 1 to be written
back and retrieved by block 2.

```bash
accelerate launch scripts/train_active_world_memory.py \
  --config configs/active_world_memory_stage3_policy.json \
  --resume /path/to/stage2/checkpoint-iter-XXXXXXXX
```

## Data audit

```bash
python scripts/debug_active_world_memory_data.py \
  --config configs/active_world_memory_stage1_retriever.json \
  --item_name zx8lnpzDG58_012000_017000.mp4
```

It writes memory/target contact sheets and a camera NPZ without loading any
model.

## Inference and dynamic memory

```bash
python scripts/inference_active_world_memory.py \
  --checkpoint /path/to/checkpoint-iter-XXXXXXXX \
  --item_name zx8lnpzDG58_012000_017000.mp4 \
  --output_dir /path/to/result
```

Each generated latent chunk is appended as a new episode before the next chunk
is retrieved.  Outputs include generated/GT/comparison videos and
`retrieval_trace.json`, which records selected source indices, query types,
regions, STOP probabilities, and final uncertainty.

For the actual capture-trajectory-to-novel-trajectory setting, pass a camera
NPZ instead of sampling a held-out target from the room tour:

```bash
python scripts/inference_active_world_memory.py \
  --checkpoint /path/to/checkpoint-iter-XXXXXXXX \
  --item_name capture_scene.mp4 \
  --target_camera_npz /path/to/novel_trajectory.npz \
  --output_dir /path/to/result
```

The NPZ must contain `c2w` (or `poses`) with shape `(T,3,4)`/`(T,4,4)` and
`intrinsics` as one or `T` camera matrices (four-vectors are also accepted).
Optional `times` are relative to the capture origin.  By default, `c2w` is in
the same ViPE/SLAM world coordinates as the capture and is normalized by the
capture first frame; use `--target_poses_normalized` only for already-relative
poses.  Arbitrary trajectory lengths are windowed with one-frame overlap so
each Wan chunk remains `1 + 4k` frames.  Without target RGB, only
`generated.mp4` and the retrieval trace are written.

Generated entries carry separate provenance and confidence.  Their retrieval
logits are confidence-weighted, their Canvas update is gated by confidence, and
only generated entries are compacted when the configured dynamic budget is
exceeded.  Compaction keeps both recent context and high-confidence older
observations; real capture entries are never evicted by this policy.

## Multi-trajectory consistency

The critic predicts one target world feature from independently sampled
evidence sets, and a consistency loss requires those predictions to agree.
This directly regularizes evidence-invariant target content.  Dynamic latent
writeback additionally forces later chunks to coexist with earlier generated
observations.  These constraints are useful for downstream 3DGS lifting while
remaining independent of a fragile globally rendered point cloud.
