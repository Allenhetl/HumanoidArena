# SONIC Action Scaling Fix

## 问题描述

当前模型可以推理，但是G1的动作异常。经过排查，发现两个关键问题：

1. **错误的动作缩放系数**: 代码使用固定的 `0.25` 作为所有关节的缩放系数
2. **错误的 last_actions 跟踪**: 代码使用当前关节位置而不是实际发送的动作

## 根本原因

### 1. 动作缩放系数错误

**错误代码** (line 622):
```python
target_sonic = raw_sonic * 0.25 + self._sonic_default_np
```

**问题**:
- 使用固定的 `0.25` 对所有29个关节进行缩放
- 但实际上每个关节应该有不同的缩放系数，取决于电机类型和力矩限制

**正确实现** (参考 `policy_parameters.hpp` lines 109-139):
```cpp
const std::array<double, 29> g1_action_scale = {
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  // left_hip_pitch
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  // left_hip_roll
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  // left_hip_yaw
    // ... 29 different values
};
```

**公式**: `action_scale = 0.25 * effort_limit / stiffness`

**电机类型**:
- **7520_22**: 大力矩电机 (髋关节、膝关节) → scale ≈ 0.351
- **7520_14**: 中力矩电机 (腰部yaw、髋部yaw) → scale ≈ 0.548
- **5020**: 标准电机 (肩部、肘部、踝关节) → scale ≈ 0.439
- **4010**: 小力矩电机 (手腕) → scale ≈ 0.075

### 2. last_actions 跟踪错误

**错误代码** (line 636):
```python
self._last_action_hist[-1] = joint_pos_sonic  # 用当前关节位置作为last action
```

**问题**:
- 使用当前关节位置 (`joint_pos_sonic`) 作为 last action
- 但 `last_action` 应该是策略网络输出的原始动作（decoder 输出，未缩放）

**正确实现** (参考 `g1_deploy_onnx_ref.cpp` lines 269, 2825):
```cpp
// 在 decoder 推理后，保存原始动作
last_action = decoder_output;  // raw action, not scaled
```

## 修复方案

### 修复 1: 添加正确的动作缩放系数

在 `action_provider_sonic.py` 中添加 `G1_ACTION_SCALE_ISAACLAB` 数组（SONIC IsaacLab order）:

```python
G1_ACTION_SCALE_ISAACLAB = np.array([
    0.3506614566,  # 0: left_hip_pitch_joint
    0.3506614566,  # 1: right_hip_pitch_joint
    0.5475464463,  # 2: waist_yaw_joint
    0.3506614566,  # 3: left_hip_roll_joint
    0.3506614566,  # 4: right_hip_roll_joint
    0.4385773242,  # 5: waist_roll_joint
    0.5475464463,  # 6: left_hip_yaw_joint
    0.5475464463,  # 7: right_hip_yaw_joint
    0.4385773242,  # 8: waist_pitch_joint
    0.3506614566,  # 9: left_knee_joint
    0.3506614566,  # 10: right_knee_joint
    0.4385773242,  # 11: left_shoulder_pitch_joint
    0.4385773242,  # 12: right_shoulder_pitch_joint
    0.4385773242,  # 13: left_ankle_pitch_joint
    0.4385773242,  # 14: right_ankle_pitch_joint
    0.4385773242,  # 15: left_shoulder_roll_joint
    0.4385773242,  # 16: right_shoulder_roll_joint
    0.4385773242,  # 17: left_ankle_roll_joint
    0.4385773242,  # 18: right_ankle_roll_joint
    0.4385773242,  # 19: left_shoulder_yaw_joint
    0.4385773242,  # 20: right_shoulder_yaw_joint
    0.4385773242,  # 21: left_elbow_joint
    0.4385773242,  # 22: right_elbow_joint
    0.4385773242,  # 23: left_wrist_roll_joint
    0.4385773242,  # 24: right_wrist_roll_joint
    0.0745008737,  # 25: left_wrist_pitch_joint
    0.0745008737,  # 26: right_wrist_pitch_joint
    0.0745008737,  # 27: left_wrist_yaw_joint
    0.0745008737,  # 28: right_wrist_yaw_joint
], dtype=np.float32)
```

**注意**: 这个数组已经从 MuJoCo order 转换为 SONIC IsaacLab order。

### 修复 2: 使用 per-joint 缩放

修改 line 659:
```python
# 修改前
target_sonic = raw_sonic * 0.25 + self._sonic_default_np

# 修改后
target_sonic = raw_sonic * G1_ACTION_SCALE_ISAACLAB + self._sonic_default_np
```

### 修复 3: 正确跟踪 last_actions

修改 lines 635-636 和 654-656:
```python
# 修改前 (line 636)
self._last_action_hist[-1] = joint_pos_sonic  # WRONG

# 修改后 (在 decoder 推理后，line 654-656)
# 更新 last_action_hist with raw action (before scaling)
self._last_action_hist = np.roll(self._last_action_hist, -1, axis=0)
self._last_action_hist[-1] = raw_sonic  # raw decoder output
```

## 验证方法

### 1. 检查动作缩放系数

运行 `compute_action_scales.py` 验证缩放系数计算正确:
```bash
cd isaaclab_twist2_g1/pico_server
python compute_action_scales.py
```

期望输出:
```
Min scale: 0.074501  (手腕关节)
Max scale: 0.547546  (腰部yaw、髋部yaw)
Mean scale: 0.381443
Unique values: 4  (4种电机类型)
```

### 2. 运行完整系统

**Terminal 1: Pico服务器**
```bash
cd GR00T-WholeBodyControl
python gear_sonic/scripts/pico_manager_thread_server.py \
    --manager --port 5556 --wbc_version sonic_model12
```

**Terminal 2: Isaac Lab仿真**
```bash
cd isaaclab_twist2_g1
bash run_sonic.sh
```

### 3. 观察日志输出

查看是否有以下日志:
```
[SONIC] Raw sonic range: [-X.XX, X.XX]  # decoder 原始输出
[SONIC] ✓ Final target range: [-X.XX, X.XX]  # 缩放后的目标
```

**预期行为**:
- Raw sonic 范围应该在 [-3, 3] 左右（策略网络输出）
- Final target 范围应该在关节限位内（例如 [-2, 2]）
- 机器人动作应该平滑、自然，跟随VR全身姿态

## 技术细节

### 动作缩放公式

```
target_joint_angle = raw_action * action_scale + default_angle
```

其中:
- `raw_action`: decoder 输出的原始动作（无单位，通常在 [-3, 3] 范围）
- `action_scale`: 每个关节的缩放系数（弧度/动作单位）
- `default_angle`: 默认站立姿态的关节角度（弧度）
- `target_joint_angle`: 最终发送给机器人的目标角度（弧度）

### 为什么需要 per-joint 缩放？

1. **不同电机有不同的力矩能力**: 大关节（髋、膝）使用大力矩电机，小关节（手腕）使用小力矩电机
2. **不同的刚度设置**: 刚度 = armature × ω²，不同电机有不同的 armature
3. **不同的力矩限制**: effort_limit 根据电机规格设定
4. **保证安全性**: 缩放系数确保动作不会超出电机能力范围

### 关节顺序说明

- **MuJoCo order**: C++ 代码中 `g1_action_scale` 数组的顺序
- **IsaacLab order**: Python 代码中 `G1_ACTION_SCALE_ISAACLAB` 的顺序
- **转换**: 使用 `mujoco_to_isaaclab` 映射进行转换

## 修复文件

- `action_provider_sonic.py`: 添加 `G1_ACTION_SCALE_ISAACLAB`，修改动作缩放和 last_actions 跟踪
- `compute_action_scales.py`: 计算和验证动作缩放系数的工具脚本

## 参考文档

1. `policy_parameters.hpp` (lines 109-139): C++ 参考实现中的 `g1_action_scale` 数组
2. `g1_deploy_onnx_ref.cpp` (line 2824): 动作缩放公式
3. `g1_deploy_onnx_ref.cpp` (lines 269, 2825): last_action 跟踪

## 修复完成时间

2026-03-04

## 测试状态

✓ 动作缩放系数计算正确
✓ last_actions 跟踪逻辑修复
⏳ 等待完整系统测试验证