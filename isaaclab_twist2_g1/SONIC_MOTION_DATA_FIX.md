# SONIC 动作不跟随问题 - 关键修复

## 🎯 问题描述

Isaac Lab 中的 SONIC 推理有输出，但机器人动作完全不正常，且不跟随 Pico VR 输入变化。

## 🔍 根本原因

**Encoder 输入数据被错误地全部清零！**

### 对比分析

#### ❌ 修复前（Isaac Lab Python 版本）
```python
# 第 1426-1433 行：所有 motion 数据被清零
motion_joint_pos_step5_full = np.zeros((_STEP5_FRAMES * 29,), dtype=np.float32)  # ❌
motion_joint_vel_step5_full = np.zeros((_STEP5_FRAMES * 29,), dtype=np.float32)  # ❌
motion_anchor_orient = np.zeros((6,), dtype=np.float32)                          # ❌
motion_anchor_orient_step5_full = np.zeros((_STEP5_FRAMES * 6,), dtype=np.float32) # ❌
motion_joint_pos_lowerbody_full = np.zeros((...), dtype=np.float32)             # ❌
motion_joint_vel_lowerbody_full = np.zeros((...), dtype=np.float32)             # ❌
```

**日志输出：**
```
[SONIC][ENCODER_BLOCKS] motion_pos_step5=[0.0000, 0.0000]  # ❌ 全为0
                        motion_vel_step5=[0.0000, 0.0000]  # ❌ 全为0
                        anchor=[0.0000, 0.0000]            # ❌ 全为0
                        lowerbody_pos=[0.0000, 0.0000]     # ❌ 全为0
```

#### ✅ 修复后（使用实际数据）
```python
# 使用从 ZMQ 接收的 joint_pos/joint_vel 数据
motion_joint_pos_step5_full = motion_joint_pos_step5_ref.reshape(-1).astype(np.float32)  # ✓
motion_joint_vel_step5_full = motion_joint_vel_step5_ref.reshape(-1).astype(np.float32)  # ✓

# 使用 SMPL anchor orientation
motion_anchor_orient = self._body_rot6d_buf[-1].copy()  # ✓
motion_anchor_orient_step5_full = gather_temporal_window(
    self._motion_anchor_rot6d_hist, _STEP5_FRAMES, _STEP5_STRIDE
).reshape(-1).astype(np.float32)  # ✓

# 使用下半身关节数据
motion_joint_pos_lowerbody_full = motion_joint_pos_lowerbody_ref.reshape(-1).astype(np.float32)  # ✓
motion_joint_vel_lowerbody_full = motion_joint_vel_lowerbody_ref.reshape(-1).astype(np.float32)  # ✓
```

**预期日志输出：**
```
[SONIC][ENCODER_BLOCKS] motion_pos_step5=[-0.6905, 0.7588]  # ✓ 有数据
                        motion_vel_step5=[...]              # ✓ 有数据
                        anchor=[-0.8282, 0.4905]            # ✓ 有数据
                        lowerbody_pos=[...]                 # ✓ 有数据
```

#### ✅ 原版 C++ (参考)
C++ 版本从 ZMQ 消息中的 `joint_pos` 和 `joint_vel` 字段填充这些数据，从不清零。

## 🔧 修复内容

### 文件：action_provider/action_provider_sonic.py

**修改位置：第 1426-1442 行**

```python
# ✨ CRITICAL FIX: Use reference motion data from ZMQ instead of zeros
# In SMPL mode, the encoder still needs robot proprioception for better tracking
motion_joint_pos_step5_full = motion_joint_pos_step5_ref.reshape(-1).astype(np.float32)
motion_joint_vel_step5_full = motion_joint_vel_step5_ref.reshape(-1).astype(np.float32)

# Extract root z position from motion history
motion_root_z_step5 = np.zeros((_STEP5_FRAMES,), dtype=np.float32)  # TODO: extract from motion if available
motion_root_z = np.zeros((1,), dtype=np.float32)

# Use anchor orientation from SMPL data
motion_anchor_orient = self._body_rot6d_buf[-1].copy()  # Latest anchor orientation
motion_anchor_orient_step5_full = gather_temporal_window(
    self._motion_anchor_rot6d_hist, _STEP5_FRAMES, _STEP5_STRIDE
).reshape(-1).astype(np.float32)

motion_joint_pos_lowerbody_full = motion_joint_pos_lowerbody_ref.reshape(-1).astype(np.float32)
motion_joint_vel_lowerbody_full = motion_joint_vel_lowerbody_ref.reshape(-1).astype(np.float32)
```

## 📊 Encoder 输入维度验证

### 完整的 1762 维 Encoder 输入

| 字段名 | 维度 | 来源 | 状态 |
|--------|------|------|------|
| encoder_mode_4 | 4 | 固定 [2, 0, 0, 0] | ✓ |
| motion_joint_positions_10frame_step5 | 290 | ZMQ joint_pos | ✅ 已修复 |
| motion_joint_velocities_10frame_step5 | 290 | ZMQ joint_vel | ✅ 已修复 |
| motion_root_z_position_10frame_step5 | 10 | TODO | ⚠️ 暂时为0 |
| motion_root_z_position | 1 | TODO | ⚠️ 暂时为0 |
| motion_anchor_orientation | 6 | SMPL body_quat | ✅ 已修复 |
| motion_anchor_orientation_10frame_step5 | 60 | SMPL body_quat 历史 | ✅ 已修复 |
| motion_joint_positions_lowerbody_10frame_step5 | 120 | ZMQ joint_pos (下半身) | ✅ 已修复 |
| motion_joint_velocities_lowerbody_10frame_step5 | 120 | ZMQ joint_vel (下半身) | ✅ 已修复 |
| vr_3point_local_target | 9 | SMPL vr_position | ✓ |
| vr_3point_local_orn_target | 12 | SMPL vr_orientation | ✓ |
| smpl_joints_10frame_step1 | 720 | SMPL joints | ✓ |
| smpl_anchor_orientation_10frame_step1 | 60 | SMPL body_quat | ✓ |
| motion_joint_positions_wrists_10frame_step1 | 60 | ZMQ joint_pos (手腕) | ✓ |
| **总计** | **1762** | | |

## 🚀 测试验证

### 重新运行测试

```bash
# Terminal 1: pico_server (如果还在运行，无需重启)
python pico_server/pico_server_pose_only.py --vis_vr3pt --vis_smpl

# Terminal 2: 重启 Isaac Lab
bash run_sonic.sh
```

### 预期结果

#### ✅ 启动日志应该看到：
```
[SonicActionProvider] ✓ ZMQ connected successfully
[ZMQ] Received raw data, size=6606 bytes
[ZMQ] SMPL data marked as VALID
[ZMQ][HISTORY] smpl_history_fill=10/10 smpl_valid=True
```

#### ✅ 运行日志应该看到（关键变化）：
```
[SONIC][ENCODER_BLOCKS] motion_pos_step5=[-0.6905, 0.7588]  # ✓ 不再是 [0.0, 0.0]
                        motion_vel_step5=[...]              # ✓ 不再是 [0.0, 0.0]
                        anchor=[-0.8282, 0.4905]            # ✓ 不再是 [0.0, 0.0]
                        anchor_step5=[...]                  # ✓ 不再是 [0.0, 0.0]
                        lowerbody_pos=[...]                 # ✓ 不再是 [0.0, 0.0]
                        lowerbody_vel=[...]                 # ✓ 不再是 [0.0, 0.0]
```

#### ✅ 机器人行为：
- 机器人应该开始跟随 VR 头显动作
- 动作应该流畅自然
- 延迟应该 < 100ms

## 🔍 为什么之前会清零？

### 原始设计意图（错误理解）

代码注释中提到：
```python
SMPL_MODE_ZEROED_BLOCKS = [
    "motion_joint_positions_10frame_step5",
    "motion_joint_velocities_10frame_step5",
    ...
]
```

这可能是基于以下**错误假设**：
- "SMPL 模式只需要 SMPL 数据，不需要机器人状态"
- "Encoder 会自动从 SMPL 推断机器人状态"

### 实际情况（C++ 参考实现）

C++ 版本**从不清零**这些字段，而是：
1. 从 ZMQ 消息中读取 `joint_pos` 和 `joint_vel`
2. 填充到 encoder 输入的对应位置
3. 即使在 SMPL 模式下，也保留机器人本体感知数据

### 为什么需要这些数据？

即使在 SMPL 模式下，Encoder 仍然需要：
1. **机器人当前状态** - 用于闭环控制
2. **历史轨迹** - 用于平滑和预测
3. **下半身状态** - 用于平衡和稳定性
4. **Anchor 方向** - 用于全局坐标对齐

## 📝 技术细节

### 数据流

```
Pico VR
  ↓
pico_server_pose_only.py
  ↓ ZMQ (包含 joint_pos, joint_vel, smpl_joints, body_quat_w)
  ↓
_fetch_zmq_pose()
  ↓
_apply_pose_data()
  ↓ 更新 _motion_joint_pos_hist, _motion_joint_vel_hist
  ↓
_run_gear_sonic()
  ↓ 从历史缓冲区采样 (step5)
  ↓ 构建 1762 维 encoder 输入
  ↓
Encoder (ONNX)
  ↓ 64 维 latent
  ↓
Decoder (ONNX)
  ↓ 29 DOF 关节目标
  ↓
Isaac Lab 机器人
```

### 关键缓冲区

```python
# 从 ZMQ 接收的参考运动数据
self._motion_joint_pos_hist  # (46, 29) - step5 采样需要 46 帧
self._motion_joint_vel_hist  # (46, 29)
self._motion_anchor_rot6d_hist  # (46, 6)

# 从 Isaac Lab 读取的当前机器人状态
self._robot_joint_pos_hist  # (10, 29) - step1 连续采样
self._robot_joint_vel_hist  # (10, 29)
```

## ✅ 修复总结

1. **问题**: Encoder 输入的 motion 数据被错误清零
2. **原因**: 误解了 SMPL 模式的数据需求
3. **修复**: 使用 ZMQ 消息中的 joint_pos/joint_vel 填充这些字段
4. **影响**: 机器人现在应该能正确跟随 VR 输入

---

**修复时间**: 2026-03-21
**修复文件**: action_provider/action_provider_sonic.py (第 1426-1442 行)
**状态**: ✅ 已修复，待测试验证
