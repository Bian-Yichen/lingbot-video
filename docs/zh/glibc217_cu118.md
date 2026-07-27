# glibc 2.17 / CUDA 11.8 兼容推理

该兼容分支面向 CentOS 7 一类 `glibc 2.17` 集群，目标环境为：

- Python 3.10
- PyTorch 2.6.0 + CUDA 11.8
- Diffusers 0.39.0
- Transformers 5.8.1
- 单卡 Diffusers backend
- Dense 1.3B 的 T2V、TI2V、T2I

高级能力（FSDP2、context parallel、SGLang native）不属于该兼容路径。

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

兼容补丁由 `lingbot_video` 包初始化，因此手工测试时也要先导入它：

```bash
python - <<'PY'
import torch
import lingbot_video
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

正常运行 `scripts/inference.py` 时会自动先加载 `lingbot_video`，无需额外处理导入顺序。

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

较旧 PyTorch 的 custom-op schema 推断不会稳定地解析由
`from __future__ import annotations` 产生的字符串类型注解。Diffusers 0.39
和 Transformers 5.8 中的部分 custom op 因此可能在导入阶段报：

```text
infer_schema(func): Parameter q has unsupported type torch.Tensor
```

`lingbot_video.compat` 只在 PyTorch 2.6 及更旧版本上，将这些类型注解解析成真实 Python 类型后再调用 PyTorch 原始的 `infer_schema`。它不修改算子实现、模型权重或采样数学过程。

单卡脚本还做了两项降级：

1. Qwen3-VL text encoder 使用 PyTorch SDPA，而不是 FlashAttention 3。
2. Transformer 的普通和 packed attention 均使用 PyTorch SDPA，不依赖
   FlashAttention 3；兼容脚本仍默认关闭 `--batch_cfg` 以降低显存峰值。

因此生成速度可能低于专用 fused attention kernel，但不需要编译额外 CUDA 扩展，
更适合老系统直接运行。
