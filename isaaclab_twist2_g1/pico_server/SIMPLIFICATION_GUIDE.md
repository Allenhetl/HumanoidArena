# Pico Server Sonic 精简指南

## 目标
只保留 POSE 模式用于 VR 全身遥操作，移除其他模式（PLANNER, PLANNER_FROZEN_UPPER_BODY, PLANNER_VR_3PT等）

## 需要保留的部分

### 1. 核心类和函数
- `_compute_rel_transform()` - 坐标转换
- `_process_3pt_pose()` - 3点姿态提取
- `process_smpl_joints()` - SMPL关节处理
- `_quat_lerp_normalized()` - 四元数插值
- `_interp_pose_axis_angle()` - 姿态插值
- `PicoReader` - Pico数据读取
- `ThreePointPose` - 3点姿态校准
- `PoseStreamer` - Pose流式传输（核心类）
- `init_hand_ik_solvers()` - 手部IK求解器初始化
- `get_controller_inputs()` - 控制器输入
- `compute_hand_joints_from_inputs()` - 手部关节计算

### 2. 需要移除的部分
- `LocomotionMode` 枚举 - 运动模式（不需要）
- `StreamMode` 枚举 - 简化为只有 POSE 模式
- `YawAccumulator` - 偏航累加器（planner模式用）
- `FeedbackReader` - 反馈读取器（planner模式用）
- `PlannerStreamer` - Planner流式传输（不需要）
- `run_pico_manager()` - 管理器模式（简化为单线程）
- 所有 planner 相关的代码

### 3. 简化后的主函数
```python
def run_pico_pose_only(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
):
    """运行Pico pose模式（仅全身遥操作）"""
    # 初始化Pico SDK
    # 创建ZMQ socket
    # 创建PoseStreamer
    # 主循环：读取Pico数据 -> 处理 -> 发送
```

### 4. 命令行参数简化
保留：
- `--buffer_size`
- `--port`
- `--num_frames_to_send`
- `--target_fps`
- `--cuda`
- `--record_dir`
- `--vis_vr3pt`
- `--no_g1`
- `--waist_tracking`
- `--vis_smpl`

移除：
- `--manager`
- `--zmq_feedback_host`
- `--zmq_feedback_port`
- `--vr3pt_test`
- `--vr3pt_live`
- `--vr3pt_realtime`

## 精简步骤

1. **复制原文件**
   ```bash
   cp pico_server_sonic.py pico_server_sonic_pose_only.py
   ```

2. **删除不需要的类**
   - 删除 `LocomotionMode` 类（line 100-123）
   - 删除 `StreamMode` 类，或简化为只有 POSE
   - 删除 `YawAccumulator` 类（line 543-582）
   - 删除 `FeedbackReader` 类（line 1576-1639）
   - 删除 `PlannerStreamer` 类（line 1641-1825）

3. **删除不需要的函数**
   - 删除 `run_pico_manager()` 函数（line 1827-2077）
   - 删除所有 planner 相关的测试函数

4. **简化 `run_pico()` 函数**
   - 移除 planner 相关的初始化
   - 移除模式切换逻辑
   - 只保留 pose streaming 部分

5. **简化 `PoseStreamer` 类**
   - 移除 planner 相关的代码
   - 移除模式切换逻辑

6. **简化 main 函数**
   - 移除 manager 模式
   - 移除测试模式
   - 只保留 pose 模式

## 核心数据流（保留）

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
    └─ vr_3pt_pose (3, 7)  # L-Wrist, R-Wrist, Neck
    ↓
PoseStreamer.run_once()
    ├─ 插值处理
    ├─ 手部IK求解
    └─ 打包数据
    ↓
pack_pose_message()
    └─ ZMQ发送
```

## 预期文件大小
- 原文件：~2200行
- 精简后：~800-1000行（减少约50-60%）

## 使用方式
```bash
# 基本用法
python pico_server_sonic_pose_only.py

# 带可视化
python pico_server_sonic_pose_only.py --vis_vr3pt --vis_smpl

# 录制数据
python pico_server_sonic_pose_only.py --record_dir ./recordings
```