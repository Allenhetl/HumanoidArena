# Pico Server Pose Only - 快速使用指南

## 快速开始

### 1. 基本运行
```bash
cd /Users/taowenwang/PycharmProjects_Simulation/HumanoidArena/isaaclab_twist2_g1/pico_server
python pico_server_pose_only.py
```

### 2. 带可视化运行（推荐）
```bash
python pico_server_pose_only.py --vis_vr3pt --vis_smpl
```

### 3. 完整参数示例
```bash
python pico_server_pose_only.py \
    --port 5556 \
    --target_fps 50 \
    --buffer_size 15 \
    --num_frames_to_send 5 \
    --vis_vr3pt \
    --vis_smpl \
    --waist_tracking \
    --cuda \
    --record_dir ./recordings
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 5556 | ZMQ发布端口 |
| `--target_fps` | 50 | 目标帧率 |
| `--buffer_size` | 15 | 滑动窗口缓冲区大小 |
| `--num_frames_to_send` | 5 | 每次发送的帧数 |
| `--vis_vr3pt` | False | 启用VR 3点姿态可视化 |
| `--vis_smpl` | False | 启用SMPL身体关节可视化 |
| `--waist_tracking` | False | 启用G1腰部追踪VR头部方向 |
| `--no_g1` | False | 禁用G1机器人可视化 |
| `--cuda` | False | 使用CUDA加速 |
| `--record_dir` | "" | 录制数据保存目录 |
| `--record_format` | "npz" | 录制格式（npz或bin） |

## 常见使用场景

### 场景1：开发调试（带可视化）
```bash
python pico_server_pose_only.py --vis_vr3pt --vis_smpl
```
- 实时查看VR 3点姿态
- 查看SMPL身体关节
- 适合调试和验证

### 场景2：生产环境（无可视化，高性能）
```bash
python pico_server_pose_only.py --target_fps 60
```
- 无可视化开销
- 更高帧率
- 适合实际遥操作

### 场景3：数据录制
```bash
python pico_server_pose_only.py \
    --record_dir ./data/recordings_$(date +%Y%m%d_%H%M%S) \
    --record_format npz
```
- 自动保存所有发送的数据
- 按时间戳命名目录
- 用于数据集收集

### 场景4：CUDA加速（GPU可用时）
```bash
python pico_server_pose_only.py --cuda --target_fps 60
```
- 使用GPU加速SMPL计算
- 更高帧率
- 需要CUDA环境

## 输出说明

### 终端输出
```
======================================================================
Pico SMPL Server - POSE Mode Only
VR Whole-body Teleoperation
======================================================================
Port: 5556
Target FPS: 50
Buffer size: 15
Frames to send: 5
VR 3pt visualization: True
SMPL visualization: True
G1 robot: True
Waist tracking: False
======================================================================
Waiting for body tracking data...
ZMQ socket bound to port 5556
[PoseStreamer] FPS: 49.87, Step: 2493
```

### ZMQ输出数据
- **话题**: `pose`
- **端口**: 5556 (可配置)
- **格式**: ZMQ packed message
- **频率**: 50 Hz (可配置)

### 数据字段
```python
{
    "smpl_pose": np.ndarray,        # (N, 63) - SMPL姿态
    "smpl_joints": np.ndarray,      # (N, 24, 3) - SMPL关节
    "body_quat_w": np.ndarray,      # (N, 24, 4) - 身体四元数
    "joint_pos": np.ndarray,        # (N, 29) - G1关节位置
    "joint_vel": np.ndarray,        # (N, 29) - G1关节速度
    "vr_position": np.ndarray,      # (9,) - 3点位置
    "vr_orientation": np.ndarray,   # (12,) - 3点四元数
    "frame_index": np.ndarray,      # (N,) - 帧索引
    "left_hand_joints": np.ndarray, # (7,) - 左手关节
    "right_hand_joints": np.ndarray,# (7,) - 右手关节
    # ... 其他字段
}
```

## 控制器操作

### Pico控制器按钮
- **扳机键（Trigger）**: 控制手部抓取
- **握把键（Grip）**: 控制手部握紧
- **菜单键**: 数据录制控制
- **摇杆**: 手部姿态微调

## 故障排除

### 问题1：找不到Pico设备
```
Error: xrobotoolkit_sdk not available
```
**解决方案**:
```bash
# 安装Pico SDK
pip install xrobotoolkit_sdk
# 或检查SDK路径
export PYTHONPATH=$PYTHONPATH:/path/to/XRoboToolkit
```

### 问题2：ZMQ端口被占用
```
Error: Address already in use
```
**解决方案**:
```bash
# 使用不同端口
python pico_server_pose_only.py --port 5557
```

### 问题3：可视化窗口无法打开
```
Warning: VR3PtPoseVisualizer not available
```
**解决方案**:
```bash
# 安装pyvista
pip install pyvista
```

### 问题4：CUDA错误
```
Error: CUDA out of memory
```
**解决方案**:
```bash
# 不使用CUDA
python pico_server_pose_only.py  # 移除 --cuda 参数
```

## 性能优化建议

1. **提高帧率**:
   - 禁用可视化: 移除 `--vis_vr3pt` 和 `--vis_smpl`
   - 使用CUDA: 添加 `--cuda` (需要GPU)
   - 提高目标FPS: `--target_fps 60`

2. **降低延迟**:
   - 减少缓冲区: `--buffer_size 5`
   - 减少发送帧数: `--num_frames_to_send 3`

3. **稳定性优先**:
   - 增加缓冲区: `--buffer_size 20`
   - 降低帧率: `--target_fps 30`

## 与C++端配合使用

### 启动顺序
1. 先启动Pico服务器（Python）
```bash
python pico_server_pose_only.py --vis_vr3pt
```

2. 再启动C++部署（接收端）
```bash
cd /path/to/gear_sonic_deploy
bash deploy.sh sim --input-type manager
```

### 验证连接
- Python端应显示: `ZMQ socket bound to port 5556`
- C++端应显示: `Connected to localhost:5556`
- 按Enter键切换到ZMQ streaming模式

## 文件位置

- **精简版本**: `pico_server_pose_only.py` (1596行)
- **原版本**: `pico_server_sonic.py` (2218行)
- **变更说明**: `CHANGES.md`
- **精简指南**: `SIMPLIFICATION_GUIDE.md`

## 相关文档

- [Sonic WBC官方文档](https://github.com/...)
- [Pico SDK文档](https://...)
- [ZMQ协议说明](../docs/zmq_protocol.md)