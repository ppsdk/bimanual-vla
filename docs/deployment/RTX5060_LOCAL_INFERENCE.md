# RTX 5060 本地推理

`dev/rtx5060-local-inference` 是独立的本地推理分支。RTX 5060 与 RTX 4090、Orin NX 并行部署，但不训练、不使用 ROS2，也不承担 Dashboard 或 Piper CAN 控制。机器人控制电脑继续运行现有 `rtc-client`，只把 Policy WebSocket 地址切换到 5060 主机。

```text
H200 训练 -> NAS -> RTX 5060 本地 SSD
                         │
机器人控制电脑 -- OpenPI WebSocket v1 --> RTX 5060 推理服务
```

## 权重交付

H200 从 NAS 同步数据训练；训练结束后，将已经导出的 PyTorch/LeRobot artifact 复制到 5060 的本地 SSD。实时推理不直接从 NAS 读取，避免网络存储抖动进入控制回路。可部署的 `pi` checkpoint 至少包含：

```text
<checkpoint>/
  model.safetensors
  assets/<dataset-id>/norm_stats.json
  policy_metadata.json        # 可选
```

`smolvla` checkpoint 使用 LeRobot 布局：

```text
<checkpoint>/
  config.json
  model.safetensors
```

SmolVLA 的输入 feature 必须按 Piper 数据合同训练或转换，单臂为 7D、双臂为 14D，且双臂需要顶视图、左腕、右腕三路图像。通用 `lerobot/smolvla_base` 不能直接执行 Piper。

## 安装和检查

5060 使用普通 x86_64 CUDA PyTorch，不安装 Jetson 专用 wheel，也不设置 `PYTORCH_NO_CUDA_MEMORY_CACHING`。建议把 Hugging Face 缓存放到本地 SSD：

```bash
python3 -m venv /opt/bimanual-vla-5060-venv
source /opt/bimanual-vla-5060-venv/bin/activate
# 先按 NVIDIA/CUDA 驱动选择并安装 torch/torchvision
python -m pip install -r requirements-rtx5060-pi.txt
export HF_HOME=/data/cache/huggingface

# SmolVLA uses a separate environment because its Transformers range differs:
# python -m pip install -r requirements-rtx5060-smolvla.txt
```

检查 CUDA 和 checkpoint：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bin/bimanual-vla local-policy-check \
  --checkpoint /data/models/pi0-piper-bimanual \
  --dataset-id piper-bimanual-v1 \
  --schema joint --arm-mode bimanual --arm-side both \
  --model-variant pi0 --profile rtx5060_8gb --device cuda
```

`local-policy-check` 不加载模型，检查 `model.safetensors`、norm stats、Piper 维度、相机 feature、权重大小、系统内存和 RTX 5060 profile。`pi05` 在 8GB profile 中是 experimental，只有在实际显存和延迟验收后才可启用。OpenPI pi 与 SmolVLA 使用不同 Transformers 版本，必须使用不同虚拟环境。

## 启动服务

OpenPI `pi` backend：

```bash
bin/bimanual-vla local-policy-server \
  --checkpoint /data/models/pi0-piper-bimanual \
  --dataset-id piper-bimanual-v1 \
  --schema joint --arm-mode bimanual --arm-side both \
  --model-variant pi0 --profile rtx5060_8gb \
  --openpi-root /opt/openpi --device cuda --precision bf16 --port 8000
```

SmolVLA backend：

```bash
bin/bimanual-vla local-policy-server \
  --backend smolvla \
  --checkpoint /data/models/smolvla-piper-bimanual \
  --dataset-id piper-bimanual-v1 \
  --schema joint --arm-mode bimanual --arm-side both \
  --profile rtx5060_8gb --device cuda --precision fp16 --port 8000
```

默认 `pi` backend 发布 model-side RTC metadata；SmolVLA 不启用 π0 专用 RTC。首次接入必须使用客户端 shadow-only，确认 handshake、三路图像键、action chunk 维度和 capture-to-result latency 后再开启真实执行。

## 客户端切换

5060 与 4090 使用同一 `openpi_websocket_v1` 协议：

```bash
bin/bimanual-vla rtc-client \
  --host <rtx5060-ip> --port 8000 \
  --arm-mode bimanual --arm-side both \
  --left-can can0 --right-can can1 \
  --cam-high-device auto --cam-left-wrist-device auto --cam-right-wrist-device auto \
  --instruction "pick up the cube" --hz 4 --control-hz 20
```

不要让 4090、5060 两个节点同时向同一个执行客户端发送动作。并行比较时，让其中一端保持 shadow-only，分别记录 `monitoring_data` 中的延迟和 RTC telemetry。

## 容量边界

| Profile | 推荐 | 实验性 |
| --- | --- | --- |
| `rtx5060_8gb` | `pi0`、`smolvla`（OpenPI BF16 / SmolVLA FP16） | `pi05` |

8GB 是显卡标称总显存，驱动通常显示约 7.5 GiB，可用于模型的显存还会更少。vision tower、语言模型、action expert、KV/cache 和预处理都会占用空间；不要把未经本机实测的 FPS 或控制频率写成保证值。默认关闭 `torch.compile`，稳定后才考虑显式指定编译模式。
