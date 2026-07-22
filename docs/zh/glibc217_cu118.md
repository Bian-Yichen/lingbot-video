# glibc 2.17 / CUDA 11.8 兼容推理

该兼容分支面向 CentOS 7 一类 `glibc 2.17` 集群，目标环境为：

- Python 3.10
- PyTorch 2.6.0 + CUDA 11.8
- Diffusers 0.39.0
- Transformers 5.8.1
- 单卡 Diffusers backend
- Dense 1.3B 的 T2V、TI2V、T2I

高级能力（FSDP2、context parallel、SGLang native、packed/batched CFG、FlashAttention 3）不属于该兼容路径。

## 1. 切换分支

```bash
git fetch origin
git checkout compat/glibc217-torch26
git pull origin compat/glibc217-torch26
```

## 2. 在现有 Conda 环境中安装

例如继续使用已有的 `3d` 环境：

```bash
conda activate 3d
bash scripts/setup_glibc217_cu118.sh
```

脚本会安装：

```text
torch==2.6.0+cu118
torchvision==0.21.0+cu118
diffusers==0.39.0
transformers==5.8.1
peft==0.19.1
```

它不会安装 Torch 2.12、CUDA 13 或要求 `manylinux_2_28` 的 PyTorch wheel。

## 3. 检查导入

```bash
python - <<'PY'
import torch
import diffusers
import transformers
from lingbot_video.pipeline_lingbot_video import LingBotVideoPipeline

print("torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("LingBot import: OK")
PY
```

## 4. 运行官方 T2V 示例

模型已经通过 Hugging Face 默认缓存下载时：

```bash
export MODEL_DIR=$(hf download robbyant/lingbot-video-dense-1.3b)
```

运行：

```bash
MODEL_DIR="$MODEL_DIR" \
OUT_DIR=outputs/dense_t2v_glibc217 \
srun -p vcg --gres=gpu:1 --quotatype=auto \
  bash scripts/single-gpu/run_dense_t2v.sh
```

输出：

```text
outputs/dense_t2v_glibc217/t2v.mp4
```

## 兼容改动说明

PyTorch 2.6 的 custom-op schema 推断不会先解析由
`from __future__ import annotations` 产生的字符串类型注解。Diffusers 0.39
和 Transformers 5.8 中的部分 custom op 因此会在导入阶段报：

```text
infer_schema(func): Parameter q has unsupported type torch.Tensor
```

`lingbot_video.compat` 只在 PyTorch 2.6 及更旧版本上，将这些类型注解解析成真实 Python 类型后再调用 PyTorch 原始的 `infer_schema`。它不修改算子实现、模型权重或采样数学过程。

单卡脚本还做了两项降级：

1. Qwen3-VL text encoder 使用 PyTorch SDPA，而不是 FlashAttention 3。
2. 不启用 `--batch_cfg`，CFG 改为正负条件各运行一次 Transformer，避免 B>1 的 packed FlashAttention 路径。

因此生成速度会比官方 Torch 2.12 + FlashAttention 3 环境慢，但更适合老系统快速跑通模型。
