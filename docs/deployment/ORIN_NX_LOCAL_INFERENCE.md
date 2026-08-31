# Orin NX 本地推理

`dev/orin-nx-local-inference` 是与 `dev/4090-ws-optimization` 并行的边缘推理分支。它不修改 `master`，也不使用 ROS2：机器人控制电脑继续运行现有 RTC client，只把 OpenPI WebSocket 的目标地址改为 Orin NX。

NX 端支持两个 backend：

- `pi`：OpenPI PyTorch checkpoint，保留现有 π0/π0.5 + RTC 路径；
- `smolvla`：LeRobot SmolVLA（约 450M 参数）路径，面向 Jetson 统一内存和 FP16；当前只支持 Piper `joint` schema，输出裁剪为单臂 7D 或双臂 14D，且不宣称支持 π0 专用 RTC。

```text
机器人控制电脑
  bimanual-vla rtc-client --host <orin-nx-ip>
             │ OpenPI WebSocket v1
             ▼
Orin NX
  bimanual-vla local-policy-server ...
             │
             ▼
  本地 model.safetensors + norm_stats
```

## 权重和数据

H200 只负责训练。训练数据从 NAS 同步到 H200，训练完成后把最终的 PyTorch checkpoint（`model.safetensors` 加 `assets/<dataset-id>/norm_stats.json`）下载到 NX 本地 SSD。NX 推理不直接从 NAS 读取，也不需要 H200 的 Python 环境或库。建议使用 `rsync`/`scp` 同步后再切换模型，避免 NAS 短暂抖动进入实时控制路径。

一个可部署 checkpoint 至少应包含：

```text
<checkpoint>/
  model.safetensors
  assets/<dataset-id>/norm_stats.json
  policy_metadata.json        # 可选；CLI 合同字段优先
```

`pi` backend 只有 `model.safetensors` 会被 edge server 接受，且需要单独的 `norm_stats.json`。只有 `params/` 的 Orbax/JAX 训练目录会在启动前明确拒绝，防止 NX 误把训练 checkpoint 当作推理 artifact。

`smolvla` backend 使用 LeRobot artifact：

```text
<checkpoint>/
  config.json
  model.safetensors
```

SmolVLA 的归一化 buffer 随 policy 一起保存在 safetensors 中，不读取 OpenPI 的 `assets/<dataset-id>` norm stats。
官方 `lerobot/smolvla_base` 是 SO-ARM100/Aloha 形态的通用权重，不能直接当作 Piper 权重执行；必须使用 Piper 7D/14D action/state 训练或转换后的 artifact。`local-policy-check` 会检查 `config.json` 中声明的维度，并在不匹配时拒绝启动。

## 先检查，再启动

在 NX 上使用独立依赖清单。不要直接安装仓库根目录的 x86/4090 `requirements.txt`，尤其不要覆盖 JetPack 提供的 PyTorch：

```bash
python3 -m venv /opt/bimanual-vla-nx-venv
source /opt/bimanual-vla-nx-venv/bin/activate
# JetPack/NVIDIA Torch 已预装时跳过 torch 安装
python -m pip install -r requirements-orin-nx.txt
```

SmolVLA 的 Jetson Torch wheel 必须能执行 `sm_87`。部署前检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

在 NX 上先做不加载模型的检查：

```bash
bin/bimanual-vla local-policy-check \
  --checkpoint /data/models/pi05-piper-bimanual \
  --dataset-id piper-bimanual-v1 \
  --schema joint \
  --arm-mode bimanual \
  --arm-side both \
  --model-variant pi05 \
  --profile orin_nx_16gb
```

检查会验证权重格式、norm stats 路径、profile 与模型系列，并打印最终 WebSocket metadata。它不导入 OpenPI、JAX 或 CUDA 模型。

## 启动本地 Policy server

```bash
bin/bimanual-vla local-policy-server \
  --checkpoint /data/models/pi05-piper-bimanual \
  --dataset-id piper-bimanual-v1 \
  --schema joint \
  --arm-mode bimanual \
  --arm-side both \
  --model-variant pi05 \
  --profile orin_nx_16gb \
  --openpi-root /opt/openpi \
  --device cuda \
  --port 8000
```

使用 SmolVLA 时：

```bash
bin/bimanual-vla local-policy-check \
  --backend smolvla \
  --checkpoint /data/models/smolvla-piper-joint \
  --dataset-id piper-joint \
  --schema joint \
  --arm-mode bimanual \
  --arm-side both \
  --profile orin_nx_8gb

bin/bimanual-vla local-policy-server \
  --backend smolvla \
  --checkpoint /data/models/smolvla-piper-joint \
  --dataset-id piper-joint \
  --schema joint \
  --arm-mode bimanual \
  --arm-side both \
  --profile orin_nx_8gb \
  --device cuda
```

参考 `smolvla-jetson` 的 Jetson 经验，SmolVLA 启动会设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1`，先在 CPU 建立并加载权重，再转为 FP16 和 CUDA；模型加载应早于相机/其他重资源初始化。官方仓库示例是 Orin Nano + SO-ARM100，本项目只复用其内存/精度策略，不复用 SO-ARM100 的控制器、关节范围或 MuJoCo 环境。

默认关闭 `torch.compile`，因为 Orin NX 上编译缓存和峰值内存必须实测；可在确认稳定后显式指定 `--compile-mode reduce-overhead` 或其它 OpenPI 支持的模式。默认开启与现有 RTC client 兼容的 model-side RTC metadata；在 shadow-only 阶段可以用 `--no-rtc-enabled` 做基线对比。

NX 服务端只负责推理和 WebSocket，不负责 Dashboard 授权、Piper CAN 或真实执行。真实动作仍由机器人客户端的双重安全门控制。第一次接入必须让 client 保持 shadow-only，确认 metadata、图像键、action 维度和延迟后再考虑执行。

## 容量边界

这里的 profile 是启动前的保守门槛，不是性能承诺：

| Profile | 推荐 | 实验性 | 备注 |
| --- | --- | --- | --- |
| `orin_nx_8gb` | `pi0`、`smolvla` | 无 | 不建议完整 `pi05`；SmolVLA 需 FP16 实测 |
| `orin_nx_16gb` | `pi0`、`smolvla` | `pi05` | π0.5 需要实测；可能需要量化、较低并发或更短 horizon |

显存/统一内存不仅包含权重，还包含 vision tower、语言模型、action expert、KV/cache、中间 tensor 和图像预处理。SmolVLA 的约 450M 参数量更适合 8 GB 起步，但具体吞吐仍需 NX 实测；没有实测前，不把 `pi05` 标为 supported，也不把 SmolVLA 的 9.6 FPS 直接当作 Piper 实时控制指标。

## 从 4090 切换到 NX

两端使用同一套 `openpi_websocket_v1` 协议。切换只改变 RTC client 的地址和端口：

```bash
# 4090
bin/bimanual-vla rtc-client --host 192.168.101.9 --port 8000 ...

# Orin NX
bin/bimanual-vla rtc-client --host <orin-nx-ip> --port 8000 ...
```

不要同时让两个节点对同一个执行 client 发动作。若要并行比较，先让其中一个 client 使用 shadow-only，并分别记录 `monitoring_data` 的 latency/RTC telemetry。

## 当前明确不做的事

- 不修改 `master`。
- 不在 NX 上训练，也不从 NAS 直接做实时推理。
- 不引入 ROS2；NX 与 4090 是并行的独立推理节点。
- 不把 `server_4090/openpi_single_arm.py` 作为 NX 入口，因为它包含 JAX/Flax/训练/Dashboard 依赖。
