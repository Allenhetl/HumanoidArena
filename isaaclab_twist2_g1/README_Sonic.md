# SONIC POSE 模式全身遥操作集成文档

<div align="center">
  <h3>基于 Isaac Lab 的 GEAR-SONIC 全身遥操作仿真</h3>
  <p>
    <a href="#overview">概述</a> •
    <a href="#architecture">系统架构</a> •
    <a href="#hardware">硬件要求</a> •
    <a href="#installation">安装</a> •
    <a href="#usage">使用方法</a> •
    <a href="#comparison">与 TWIST2 对比</a>
  </p>
</div>

---

## <a id="overview"></a>📖 概述

本项目将 NVIDIA GEAR-SONIC 全身遥操作系统集成到 Isaac Lab 仿真环境中，支持 **POSE 模式**的完整全身跟踪（包括腿部运动）。

### POSE 模式特性

- **完整全身跟踪**：Pico 头显 + 手腕控制器 + **脚踝 tracker** → 完整 SMPL 24 关节姿态
- **腿部运动跟踪**：通过脚踝 tracker 实现下半身运动跟踪（非 RL 自主控制）
- **GEAR-SONIC 策略**：端到端神经网络 retargeting（SMPL 24 joints → G1 29 DOF）
- **Isaac Lab 仿真**：PhysX 物理引擎，支持多任务场景

### 与 VR_3PT 模式的区别

| 特性 | VR_3PT 模式 | POSE 模式（本项目） |
|------|-------------|---------------------|
| 输入设备 | 头显 + 2 手腕控制器 | 头显 + 2 手腕 + **2 脚踝 tracker** |
| 上半身控制 | Pink IK 求解 | GEAR-SONIC 全身策略 |
| 下半身控制 | RL locomotion（自主） | **跟踪人体腿部运动** |
| SMPL 数据 | 仅上半身推算 | 完整 24 关节（含腿部） |
| 适用场景 | 固定站立操作 | 全身运动（行走、蹲下等） |

---

## <a id="architecture"></a>🏗️ 系统架构

### 数据流

```
Pico 头显 + 手腕控制器 + 脚踝 tracker (20 Hz)
    ↓
pico_manager_thread_server.py (gear_sonic 侧)
    ├─ 拟合完整 SMPL body pose (24 joints)
    ├─ smpl_joints: (N, 24, 3)  局部坐标
    ├─ smpl_pose:   (N, 21, 3)  轴角旋转
    └─ body_quat_w: (N, 4)      全局朝向
    ↓
ZMQ "pose" topic (50 Hz, Protocol v3)
    ↓
┌─────────────────────────────────────────────────┐
│  方案 A: 直接集成（推荐）                        │
│  action_provider_sonic.py                       │
│    ├─ 读取 SMPL 数据                            │
│    ├─ GEAR-SONIC encoder ONNX                   │
│    ├─ GEAR-SONIC decoder ONNX                   │
│    └─ 输出 G1 29 DOF 关节目标                   │
└─────────────────────────────────────────────────┘
    ↓
Isaac Lab 仿真 (200 Hz, 4x decimation)
    ├─ PhysX 物理引擎
    ├─ 多相机渲染
    └─ XRobot H.264 图像流
```

```
┌─────────────────────────────────────────────────┐
│  方案 B: DDS 桥接（零代码修改）                  │
│  sonic_wbc_bridge.py (独立进程)                 │
│    ├─ 读取 ZMQ SMPL 数据                        │
│    ├─ GEAR-SONIC encoder+decoder                │
│    └─ 发布到 DDS rt/lowcmd                      │
└─────────────────────────────────────────────────┘
    ↓
DDS rt/lowcmd (Unitree SDK2)
    ↓
action_provider_wh_dds.py (twist2 现有代码)
    ↓
Isaac Lab 仿真
```

### 两种集成方案对比

| 特性 | 方案 A：直接集成 | 方案 B：DDS 桥接 |
|------|------------------|------------------|
| 入口脚本 | `run_sonic.sh` | `run_sonic_dds.sh` |
| 代码修改 | 新增 `action_provider_sonic.py` | 零修改（复用 twist2 DDS） |
| 进程数量 | 2 个（Pico + Isaac Lab） | 3 个（Pico + Bridge + Isaac Lab） |
| 通信链路 | ZMQ → ONNX → Isaac | ZMQ → ONNX → DDS → Isaac |
| 性能 | **最优**（直接推理） | 较好（额外 DDS 开销） |
| 推荐场景 | 生产环境、高频控制 | 快速验证、兼容性测试 |

---

## <a id="hardware"></a>🔧 硬件要求

### 必需硬件

1. **Pico 头显**（支持 POSE 模式）
   - Pico 4 或更高版本
   - 固件版本需支持全身跟踪

2. **手腕控制器** × 2
   - Pico 原装控制器

3. **脚踝 tracker** × 2（**必须**）
   - 用于下半身运动跟踪
   - 需固定在脚踝位置
   - **穿紧身裤以保证视线**

4. **GPU**
   - NVIDIA RTX 3080 或更高
   - 推荐 RTX 4090（Isaac Sim 5.0）
   - VRAM ≥ 10GB

5. **CPU**
   - Intel i7 或 AMD Ryzen 7 以上
   - 推荐 8 核心以上

### 软件环境

- Ubuntu 20.04 / 22.04
- Isaac Sim 4.5.0 或 5.0.0
- Isaac Lab
- CUDA 11.8+ / 12.x
- Python 3.10+

---

## <a id="installation"></a>⚙️ 安装

### 1. Isaac Lab 环境

参考主 README 安装 Isaac Sim 和 Isaac Lab：
- [Isaac Sim 4.5.0 安装](doc/isaacsim4.5_install_zh.md)（RTX 4080 以下）
- [Isaac Sim 5.0.0 安装](doc/isaacsim5.0_install_zh.md)（RTX 4080 以上）

### 2. GR00T-WholeBodyControl 仓库

```bash
cd /Users/taowenwang/PycharmProjects_Simulation/HumanoidArena
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl

# 安装依赖
pip install -e .
pip install onnxruntime-gpu  # 或 onnxruntime（CPU）
```

### 3. Unitree SDK2

```bash
# Python SDK
pip install unitree_sdk2py

# C++ SDK（可选，用于 gear_sonic_deploy）
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install
```

### 4. GEAR-SONIC 模型

下载 ONNX 模型（从 GEAR-SONIC 官方发布）：

```bash
cd GR00T-WholeBodyControl/gear_sonic_deploy/policy/release
# 确保存在以下文件：
# - model_encoder.onnx
# - model_decoder.onnx
# - observation_config.yaml
```

### 5. 验证安装

```bash
cd isaaclab_twist2_g1
python -c "
from gear_sonic.utils.teleop.zmq.zmq_poller import ZMQPoller
import onnxruntime as ort
print('✓ gear_sonic 导入成功')
print('✓ onnxruntime 可用')
"
```

---

## <a id="usage"></a>🚀 使用方法

### 方案 A：直接集成（推荐）

#### 步骤 1：启动 Pico VR 数据采集

```bash
# Terminal 1
cd GR00T-WholeBodyControl
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager \
    --port 5556 \
    --wbc_version sonic_model12
```

**重要**：
- 确保 Pico 头显已连接并校准
- 佩戴 2 个脚踝 tracker
- 穿紧身裤以保证 tracker 视线
- 在 Pico 界面选择 **POSE 模式**

#### 步骤 2：启动 Isaac Lab 仿真

```bash
# Terminal 2
cd isaaclab_twist2_g1
bash run_sonic.sh
```

或手动指定模型路径：

```bash
bash run_sonic.sh \
    --encoder /path/to/model_encoder.onnx \
    --decoder /path/to/model_decoder.onnx
```

#### 步骤 3：开始遥操作

1. 在 Isaac Lab 窗口中，点击 `PerspectiveCamera → Cameras → PerspectiveCamera` 查看主视图
2. 在 Pico 头显中移动身体，机器人将跟随你的全身运动（包括腿部）
3. 按 Pico 控制器按钮控制手部抓取

### 方案 B：DDS 桥接

#### 步骤 1：启动 Pico VR（同方案 A）

```bash
# Terminal 1
cd GR00T-WholeBodyControl
python gear_sonic/scripts/pico_manager_thread_server.py --manager --port 5556
```

#### 步骤 2：启动 SONIC WBC Bridge

```bash
# Terminal 2
cd isaaclab_twist2_g1
python sonic_wbc_bridge.py \
    --zmq_port 5556 \
    --encoder ../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx \
    --decoder ../GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx \
    --domain_id 1
```

#### 步骤 3：启动 Isaac Lab

```bash
# Terminal 3
cd isaaclab_twist2_g1
bash run_sonic_dds.sh
```

---

## <a id="comparison"></a>📊 与 TWIST2 对比

### TWIST2（OpenPI + GMR）

```
第一人称相机 + 语言指令
    ↓
OpenPI0.5 策略（Transformer）
    ↓
SMPL 动作序列（16 帧）
    ↓
SMPL-X 正运动学
    ↓
GMR IK retargeting（显式优化）
    ↓
G1 29 DOF 关节目标
    ↓
TWIST2 低层跟踪策略
    ↓
Isaac Lab 执行
```

**特点**：
- 输入：视觉 + 语言（离线生成轨迹）
- Retargeting：GMR（数据驱动，快速查表）
- 适用：预定义任务、数据集生成

### SONIC POSE（本项目）

```
Pico 全身 tracker（头 + 手腕 + 脚踝）
    ↓
实时 SMPL 拟合（20 Hz）
    ↓
GEAR-SONIC encoder+decoder（端到端神经网络）
    ↓
G1 29 DOF 关节目标（含腿部跟踪）
    ↓
Isaac Lab 直接执行
```

**特点**：
- 输入：实时 VR 跟踪（在线交互）
- Retargeting：GEAR-SONIC（端到端学习，物理约束）
- 适用：实时遥操作、全身运动

### 技术对比

| 维度 | TWIST2 (OpenPI + GMR) | SONIC POSE |
|------|----------------------|------------|
| **输入模态** | 视觉 + 语言 | VR 全身跟踪 |
| **实时性** | 离线生成（16 帧批处理） | 在线实时（50 Hz） |
| **腿部控制** | SMPL 全身 → GMR 全身 | SMPL 全身 → GEAR-SONIC 全身 |
| **Retargeting** | GMR（显式 IK 优化） | GEAR-SONIC（端到端神经网络） |
| **物理约束** | 后处理约束 | 策略内嵌约束 |
| **运动平滑** | TWIST2 低层策略 | GEAR-SONIC 内置平滑 |
| **硬件需求** | 相机 | Pico + 脚踝 tracker |
| **适用场景** | 数据集生成、预定义任务 | 实时遥操作、探索性任务 |

### 性能对比

| 指标 | TWIST2 | SONIC POSE |
|------|--------|------------|
| 端到端延迟 | ~200ms（OpenPI 推理） | ~50ms（GEAR-SONIC 推理） |
| 控制频率 | 50 Hz（TWIST2 策略） | 50 Hz（GEAR-SONIC 输出） |
| GPU 占用 | 高（OpenPI Transformer） | 中（ONNX encoder+decoder） |
| 腿部跟踪精度 | 高（GMR 显式优化） | 高（GEAR-SONIC 学习） |

---

## 🛠️ 故障排查

### 1. ZMQ 连接失败

**错误**：`[SonicActionProvider] ZMQ init failed: Connection refused`

**解决**：
- 确保 `pico_manager_thread_server.py` 已启动
- 检查端口是否被占用：`lsof -i :5556`
- 确认 ZMQ 端口参数一致（默认 5556）

### 2. ONNX 模型加载失败

**错误**：`[SonicActionProvider] failed to load model_encoder.onnx`

**解决**：
- 检查模型文件路径是否正确
- 确认 ONNX 模型版本与 onnxruntime 兼容
- 尝试使用 CPU provider：`export ORT_DISABLE_CUDA=1`

### 3. 脚踝 tracker 未检测到

**症状**：机器人腿部不跟随人体运动

**解决**：
- 确认 Pico 设置中已启用 POSE 模式（非 VR_3PT）
- 检查脚踝 tracker 电量和连接状态
- 穿紧身裤，确保 tracker 视线不被遮挡
- 在 Pico 界面重新校准 tracker

### 4. 仿真卡顿

**症状**：Isaac Lab 帧率低于 30 FPS

**解决**：
- 降低相机分辨率：`--image_xrobot_width 320 --image_xrobot_height 240`
- 关闭世界相机：移除 `--enable_world_camera`
- 使用 CPU provider（降低 GPU 负载）：修改 `action_provider_sonic.py` 中 `providers = ["CPUExecutionProvider"]`

### 5. 机器人姿态异常

**症状**：机器人关节角度超出范围或抖动

**解决**：
- 检查 SMPL 数据是否正常：在 `pico_manager_thread_server.py` 中打印 `smpl_joints`
- 确认 GEAR-SONIC 模型与 G1 机器人配置匹配
- 调整 action_scale（默认 0.25）：修改 `action_provider_sonic.py` line 367

---

## 📝 配置参数

### sim_main.py 参数

```bash
python sim_main.py \
    --action_source sonic_wholebody \          # 使用 SONIC action provider
    --sonic_zmq_host localhost \               # ZMQ 服务器地址
    --sonic_zmq_port 5556 \                    # ZMQ 端口
    --sonic_encoder_path /path/to/encoder.onnx \  # encoder 模型路径
    --sonic_decoder_path /path/to/decoder.onnx \  # decoder 模型路径
    --task Isaac-Move-Cylinder-G129-Dex3-Wholebody \  # 任务场景
    --robot_type g129 \                        # 机器人类型
    --enable_dex3_dds \                        # 启用 Dex3 灵巧手
    --device cuda \                            # 计算设备
    --enable_cameras \                         # 启用相机渲染
    --enable_world_camera                      # 启用第三人称相机
```

### pico_manager_thread_server.py 参数

```bash
python pico_manager_thread_server.py \
    --manager \                                # 启用 manager 模式
    --port 5556 \                              # ZMQ 发布端口
    --wbc_version sonic_model12                # WBC 版本
```

---

## 🎯 使用场景

### 1. 实时全身遥操作

适用于需要实时控制机器人全身运动的场景：
- 复杂环境探索
- 动态障碍物规避
- 人机协作任务

### 2. 全身运动数据采集

通过 POSE 模式采集高质量全身运动数据：
- 行走、跑步、跳跃
- 蹲下、起立、转身
- 复杂操作任务（搬运、组装）

### 3. 策略验证

验证全身控制策略的性能：
- GEAR-SONIC 策略评估
- 与其他 retargeting 方法对比
- 物理约束测试

---

## 📚 参考资料

### 论文

- **SONIC**: [arxiv.org/abs/2511.07820](https://arxiv.org/abs/2511.07820)
- **TWIST2**: 相关论文链接
- **GMR**: General Motion Retargeting

### 文档

- [GEAR-SONIC 官方文档](https://nvlabs.github.io/GR00T-WholeBodyControl/index.html)
- [Isaac Lab 文档](https://isaac-sim.github.io/IsaacLab/)
- [Unitree SDK2 文档](https://github.com/unitreerobotics/unitree_sdk2)

### 代码仓库

- [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)

---

## 🙏 致谢

本项目基于以下开源项目：

1. [NVIDIA GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
2. [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
3. [Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2)
4. [ZeroMQ](https://github.com/zeromq/pyzmq)

---

## 📄 许可证

遵循各项目的原始许可证：
- GEAR-SONIC: [LICENSE](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/LICENSE)
- Isaac Lab: BSD 3-Clause
- Unitree SDK2: Apache 2.0

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues: [HumanoidArena/issues](https://github.com/your-repo/issues)
- Email: your-email@example.com
