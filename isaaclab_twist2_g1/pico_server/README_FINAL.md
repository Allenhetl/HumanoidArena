# Pico Server Pose Only - 最终修复说明

## 修复历史

### 问题1: IndentationError (已修复 ✓)
```
IndentationError: expected an indented block after function definition on line 79
```
**原因**: 文件创建时被截断，函数定义不完整
**解决方案**: 重新从原文件提取完整的函数定义

### 问题2: NameError - threading not defined (已修复 ✓)
```
NameError: name 'threading' is not defined
```
**原因**: 缺少 `threading` 模块导入
**解决方案**: 在导入部分添加 `import threading`

### 问题3: NameError - build_command_message not defined (已修复 ✓)
```
NameError: name 'build_command_message' is not defined
```
**原因**: 只导入了 `pack_pose_message`，缺少 `build_command_message` 和 `build_planner_message`
**解决方案**: 更新导入语句，添加这两个函数，并在ImportError时设置为None

## 当前状态

✓ 所有语法错误已修复
✓ 所有必需的导入已添加
✓ 文件可以正常运行
✓ Pico设备连接正常

### 最新导入部分
```python
try:
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        pack_pose_message,
        build_command_message,
        build_planner_message,
    )
except ImportError:
    def pack_pose_message(*args, **kwargs) -> bytes:
        raise RuntimeError("pack_pose_message unavailable")
    build_command_message = None
    build_planner_message = None
```

## 使用方法

### 基本运行
```bash
python pico_server_pose_only.py
```

### 带可视化运行（推荐）
```bash
python pico_server_pose_only.py --vis_vr3pt --vis_smpl
```

### 完整参数示例
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

## 预期输出

### 正常启动
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
waiting for body data...
:/opt/apps/roboticsservice:/opt/apps/roboticsservice/lib:/opt/apps/roboticsservice/SDK/x64
/opt/apps/roboticsservice/plugins/:
/opt/apps/roboticsservice/qml/:
release mode
initialize sdk,connect127.0.0.1:60061client start server stream server connect

server connectwatch server feedback thread startdevice found
TestDevice
ZMQ socket bound to port 5556
device find TestDevice
[PoseStreamer] FPS: 49.87, Step: 2493
```

### 数据流动
当Pico设备正常工作时，你会看到：
- `device found TestDevice` - 设备已找到
- `ZMQ socket bound to port 5556` - ZMQ服务器已启动
- `[PoseStreamer] FPS: XX.XX, Step: XXXX` - 数据正在流动

## 可视化说明

### VR 3点可视化 (--vis_vr3pt)
显示3个关键点的姿态：
- 左手腕 (Left Wrist)
- 右手腕 (Right Wrist)
- 颈部 (Neck)

### SMPL可视化 (--vis_smpl)
显示24个SMPL身体关节的球体

### G1机器人可视化
默认启用，显示G1机器人模型跟随VR姿态

## 故障排除

### 问题: 等待body data超时
```
waiting for body data...
waiting for body data...
```

**可能原因**:
1. Pico设备未连接或未开机
2. 身体追踪服务未运行
3. Pico应用未启动

**解决方案**:
```bash
# 检查服务状态
ps aux | grep roboticsservice

# 重启服务
sudo systemctl restart roboticsservice
# 或
bash /opt/apps/roboticsservice/runService.sh

# 检查Pico设备连接
# 确保Pico头显已开机并运行身体追踪应用
```

### 问题: 找不到xrobotoolkit_sdk
```bash
pip install xrobotoolkit_sdk
# 或设置PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/XRoboToolkit
```

### 问题: ZMQ端口被占用
```bash
# 使用不同端口
python pico_server_pose_only.py --port 5557
```

### 问题: 可视化窗口无法打开
```bash
# 安装pyvista
pip install pyvista
```

### 问题: CUDA错误
```bash
# 不使用CUDA
python pico_server_pose_only.py  # 移除 --cuda 参数
```

## 与C++端配合使用

### 启动顺序

1. **先启动Pico服务器（Python）**
```bash
cd /path/to/isaaclab_twist2_g1/pico_server
python pico_server_pose_only.py --vis_vr3pt
```

等待看到：
```
ZMQ socket bound to port 5556
[PoseStreamer] FPS: XX.XX, Step: XXXX
```

2. **再启动C++部署（接收端）**
```bash
cd /path/to/gear_sonic_deploy
bash deploy.sh sim --input-type manager
```

3. **切换到ZMQ streaming模式**
   - 按 `Enter` 键切换到ZMQ streaming模式
   - 应该看到 "ZMQ STREAMING MODE: ENABLED"

### 验证连接
- Python端应显示: `ZMQ socket bound to port 5556`
- C++端应显示: `Connected to localhost:5556`
- 数据开始流动时会显示FPS信息

## 数据流

```
Pico设备 (身体追踪)
    ↓
xrobotoolkit_sdk
    ↓
PicoReader.read_once()
    ├─ body_pose (21, 3)
    ├─ global_orient (3,)
    └─ transl (3,)
    ↓
process_smpl_joints()
    ├─ smpl_joints_local (24, 3)
    ├─ body_quat (24, 4)
    └─ smpl_pose (63,)
    ↓
ThreePointPose.process_smpl_pose()
    └─ vr_3pt_pose (3, 7)
    ↓
PoseStreamer.run_once()
    ├─ 插值处理
    ├─ 手部IK求解
    └─ 打包数据
    ↓
pack_pose_message()
    └─ ZMQ发送 (port 5556)
    ↓
C++ gear_sonic_deploy (接收端)
    └─ 策略推理 → 机器人控制
```

## 输出数据格式

```python
{
    "smpl_pose": (N, 63),           # SMPL姿态参数
    "smpl_joints": (N, 24, 3),      # SMPL关节位置
    "body_quat_w": (N, 24, 4),      # 身体四元数
    "joint_pos": (N, 29),           # G1关节位置
    "joint_vel": (N, 29),           # G1关节速度
    "vr_position": (9,),            # 3点位置
    "vr_orientation": (12,),        # 3点四元数
    "frame_index": (N,),            # 帧索引
    "left_hand_joints": (7,),       # 左手关节
    "right_hand_joints": (7,),      # 右手关节
    "left_trigger": (1,),           # 左扳机
    "right_trigger": (1,),          # 右扳机
    "left_grip": (1,),              # 左握把
    "right_grip": (1,),             # 右握把
    # ... 其他字段
}
```

## 性能优化

### 提高帧率
```bash
# 禁用可视化
python pico_server_pose_only.py --target_fps 60

# 使用CUDA（需要GPU）
python pico_server_pose_only.py --cuda --target_fps 60
```

### 降低延迟
```bash
# 减少缓冲区和发送帧数
python pico_server_pose_only.py \
    --buffer_size 5 \
    --num_frames_to_send 3
```

### 稳定性优先
```bash
# 增加缓冲区，降低帧率
python pico_server_pose_only.py \
    --buffer_size 20 \
    --target_fps 30
```

## 版本历史

- **v1.0** - 初始精简版本（有IndentationError）
- **v1.1** - 修复IndentationError
- **v1.2** - 修复threading导入问题
- **v1.3** - 修复build_command_message和build_planner_message导入问题 ✓ (当前版本)

## 相关文档

- `CHANGES.md` - 详细变更说明
- `QUICKSTART.md` - 快速使用指南
- `SIMPLIFICATION_GUIDE.md` - 精简指南
- `test_import.py` - Python导入测试脚本
- `verify_fix.sh` - Bash验证脚本