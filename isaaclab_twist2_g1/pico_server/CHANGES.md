# Pico Server Sonic 精简版本变更说明

## 文件对比

| 项目 | 原版本 | 精简版本 | 变化 |
|------|--------|----------|------|
| 文件名 | `pico_server_sonic.py` | `pico_server_pose_only.py` | - |
| 行数 | 2218 | 1596 | -622 (-28%) |
| 功能 | 多模式（POSE/PLANNER等） | 仅POSE模式 | 简化 |

## 移除的内容

### 1. 移除的类
- ✗ `LocomotionMode` (IntEnum) - 运动模式枚举（IDLE, WALK, RUN等）
- ✗ `YawAccumulator` - 偏航角累加器（planner模式使用）
- ✗ `FeedbackReader` - ZMQ反馈读取器（planner模式使用）
- ✗ `PlannerStreamer` - Planner流式传输器

### 2. 移除的函数
- ✗ `run_pico_manager()` - 多线程管理器模式
- ✗ `run_vr3pt_visualizer_test()` - VR 3点可视化测试
- ✗ `run_vr3pt_live_visualizer()` - VR 3点实时可视化
- ✗ `run_vr3pt_realtime_visualizer()` - VR 3点实时可视化（独立模式）

### 3. 移除的命令行参数
- ✗ `--manager` - 管理器模式
- ✗ `--zmq_feedback_host` - ZMQ反馈主机
- ✗ `--zmq_feedback_port` - ZMQ反馈端口
- ✗ `--vr3pt_test` - VR 3点测试模式
- ✗ `--vr3pt_live` - VR 3点实时模式
- ✗ `--vr3pt_realtime` - VR 3点实时可视化
- ✗ `--vr3pt_hz` - VR可视化更新频率

## 保留的内容

### 1. 核心类（完整保留）
- ✓ `PicoReader` - Pico数据读取器
- ✓ `ThreePointPose` - 3点姿态处理和校准
- ✓ `PoseStreamer` - Pose流式传输器（核心）

### 2. 核心函数（完整保留）
- ✓ `_compute_rel_transform()` - 坐标系转换
- ✓ `_process_3pt_pose()` - 3点姿态提取
- ✓ `process_smpl_joints()` - SMPL关节处理
- ✓ `_quat_lerp_normalized()` - 四元数插值
- ✓ `_interp_pose_axis_angle()` - 姿态插值
- ✓ `init_hand_ik_solvers()` - 手部IK求解器初始化
- ✓ `get_controller_inputs()` - 控制器输入读取
- ✓ `get_controller_axes()` - 控制器轴读取
- ✓ `get_menu_buttons()` - 菜单按钮读取
- ✓ `get_axis_clicks()` - 轴点击读取
- ✓ `get_face_buttons()` - 面部按钮读取
- ✓ `get_abxy_buttons()` - ABXY按钮读取
- ✓ `compute_hand_joints_from_inputs()` - 手部关节计算
- ✓ `run_pico()` - 主运行函数（POSE模式）

### 3. 保留的命令行参数
- ✓ `--buffer_size` - 缓冲区大小
- ✓ `--port` - ZMQ端口
- ✓ `--num_frames_to_send` - 发送帧数
- ✓ `--target_fps` - 目标FPS
- ✓ `--cuda` - 使用CUDA
- ✓ `--record_dir` - 录制目录
- ✓ `--record_format` - 录制格式
- ✓ `--vis_vr3pt` - VR 3点可视化
- ✓ `--no_g1` - 禁用G1机器人可视化
- ✓ `--waist_tracking` - 腰部追踪
- ✓ `--vis_smpl` - SMPL可视化

## 使用方式

### 原版本（多模式）
```bash
# Manager模式（多线程，支持模式切换）
python pico_server_sonic.py --manager --vis_vr3pt

# 单线程POSE模式
python pico_server_sonic.py --vis_vr3pt
```

### 精简版本（仅POSE模式）
```bash
# 基本用法
python pico_server_pose_only.py

# 带可视化
python pico_server_pose_only.py --vis_vr3pt --vis_smpl

# 完整参数
python pico_server_pose_only.py \
    --port 5556 \
    --target_fps 50 \
    --buffer_size 15 \
    --num_frames_to_send 5 \
    --vis_vr3pt \
    --vis_smpl \
    --waist_tracking \
    --record_dir ./recordings
```

## 数据流（保持不变）

```
Pico SDK (xrobotoolkit_sdk)
    ↓
PicoReader.read_once()
    ├─ body_pose (21, 3) - SMPL姿态参数
    ├─ global_orient (3,) - 全局方向
    └─ transl (3,) - 平移
    ↓
process_smpl_joints()
    ├─ smpl_joints_local (24, 3) - SMPL局部关节
    ├─ body_quat (24, 4) - 身体四元数
    └─ smpl_pose (63,) - SMPL姿态
    ↓
ThreePointPose.process_smpl_pose()
    └─ vr_3pt_pose (3, 7) - 3点姿态（左手腕、右手腕、颈部）
    ↓
PoseStreamer.run_once()
    ├─ 插值处理
    ├─ 手部IK求解
    └─ 打包数据
    ↓
pack_pose_message()
    └─ ZMQ发送到端口5556
```

## 输出数据格式（保持不变）

```python
{
    "smpl_pose": (N, 63),           # SMPL姿态参数
    "smpl_joints": (N, 24, 3),      # SMPL关节位置
    "body_quat_w": (N, 24, 4),      # 身体四元数
    "joint_pos": (N, 29),           # G1关节位置（仅手腕有值）
    "joint_vel": (N, 29),           # G1关节速度（全零）
    "vr_position": (9,),            # 3点位置
    "vr_orientation": (12,),        # 3点四元数
    "frame_index": (N,),            # 帧索引
    "left_hand_joints": (7,),       # 左手关节
    "right_hand_joints": (7,),      # 右手关节
    # ... 其他字段
}
```

## 兼容性

- ✓ 与原版本的ZMQ输出格式完全兼容
- ✓ 可以直接替换原版本用于VR全身遥操作
- ✓ 所有POSE模式的功能保持不变
- ✗ 不支持PLANNER模式及其变体
- ✗ 不支持多线程管理器模式

## 优势

1. **代码更简洁**：减少28%的代码量
2. **更易维护**：移除了不需要的复杂模式切换逻辑
3. **专注功能**：专注于VR全身遥操作（POSE模式）
4. **性能相同**：核心功能和性能与原版本完全一致
5. **易于理解**：代码结构更清晰，更容易理解和修改

## 注意事项

1. 如果需要PLANNER模式，请使用原版本 `pico_server_sonic.py`
2. 精简版本不支持运行时模式切换
3. 所有可视化功能保持不变
4. 录制功能保持不变