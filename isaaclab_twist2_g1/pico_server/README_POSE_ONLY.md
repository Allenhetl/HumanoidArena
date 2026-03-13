# Pico Server Pose Only - 使用说明

## 文件说明

已成功创建精简版本：`pico_server_pose_only.py`

- **原文件**: `pico_server_sonic.py` (2218行)
- **精简版**: `pico_server_pose_only.py` (1571行)
- **减少**: 647行 (29%)

## 修复的问题

原始错误：
```
IndentationError: expected an indented block after function definition on line 79
```

**原因**: 文件创建时被截断，函数定义不完整

**解决方案**: 重新从原文件提取完整的函数定义

## 验证结果

✓ 语法检查通过
✓ 所有核心函数和类都存在
✓ 不需要的类已移除（LocomotionMode, FeedbackReader, PlannerStreamer等）

## 使用方法

### 1. 基本运行
```bash
cd /Users/taowenwang/PycharmProjects_Simulation/HumanoidArena/isaaclab_twist2_g1/pico_server
python pico_server_pose_only.py
```

### 2. 带可视化运行（推荐）
```bash
python pico_server_pose_only.py --vis_vr3pt --vis_smpl
```

### 3. 完整参数
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

## 可用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | 5556 | ZMQ发布端口 |
| `--target_fps` | 50 | 目标帧率 |
| `--buffer_size` | 15 | 滑动窗口缓冲区大小 |
| `--num_frames_to_send` | 5 | 每次发送的帧数 |
| `--vis_vr3pt` | False | 启用VR 3点姿态可视化 |
| `--vis_smpl` | False | 启用SMPL身体关节可视化 |
| `--waist_tracking` | False | 启用G1腰部追踪 |
| `--no_g1` | False | 禁用G1机器人可视化 |
| `--cuda` | False | 使用CUDA加速 |
| `--record_dir` | "" | 录制数据保存目录 |
| `--record_format` | "npz" | 录制格式 |

## 保留的核心功能

### 类
- ✓ `PicoReader` - Pico数据读取
- ✓ `ThreePointPose` - 3点姿态处理
- ✓ `PoseStreamer` - Pose流式传输

### 函数
- ✓ `_compute_rel_transform()` - 坐标转换
- ✓ `_process_3pt_pose()` - 3点姿态提取
- ✓ `process_smpl_joints()` - SMPL关节处理
- ✓ `init_hand_ik_solvers()` - 手部IK初始化
- ✓ `get_controller_inputs()` - 控制器输入
- ✓ `compute_hand_joints_from_inputs()` - 手部关节计算
- ✓ `run_pico()` - 主运行函数

## 移除的内容

### 类
- ✗ `LocomotionMode` - 运动模式枚举
- ✗ `YawAccumulator` - 偏航累加器
- ✗ `FeedbackReader` - 反馈读取器
- ✗ `PlannerStreamer` - Planner流式传输

### 函数
- ✗ `run_pico_manager()` - 多线程管理器
- ✗ `run_vr3pt_visualizer_test()` - 测试函数
- ✗ `run_vr3pt_live_visualizer()` - 测试函数
- ✗ `run_vr3pt_realtime_visualizer()` - 测试函数

## 数据流

```
Pico SDK (xrobotoolkit_sdk)
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
}
```

## 故障排除

### 问题：找不到xrobotoolkit_sdk
```bash
pip install xrobotoolkit_sdk
```

### 问题：ZMQ端口被占用
```bash
python pico_server_pose_only.py --port 5557
```

### 问题：可视化不可用
```bash
pip install pyvista
```

## 与C++端配合

### 启动顺序
1. 先启动Pico服务器
```bash
python pico_server_pose_only.py --vis_vr3pt
```

2. 再启动C++部署
```bash
cd /path/to/gear_sonic_deploy
bash deploy.sh sim --input-type manager
```

3. 按Enter键切换到ZMQ streaming模式

## 相关文档

- `CHANGES.md` - 详细变更说明
- `QUICKSTART.md` - 快速使用指南
- `SIMPLIFICATION_GUIDE.md` - 精简指南