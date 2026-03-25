# SONIC数据流对比分析

## 发现的关键问题

通过对比`pico_server_pose_only.py`和`action_provider_sonic.py`，发现以下潜在问题：

### 问题1：joint_pos的来源和用途不一致

**Pico Server端（发送）**：
```python
# pico_server_pose_only.py:1287-1344
joint_pos = np.zeros(29)
# 手腕部分是从SMPL的肘部+手腕旋转手动计算的
joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]
# 其他关节位置为0（未填充）
```

**IsaacLab端（接收）**：
```python
# action_provider_sonic.py:734-738
if "joint_pos" in data:
    jp = data["joint_pos"].astype(np.float32)
    self._robot_joint_pos = jp[-1]
    self._motion_joint_pos_hist = np.roll(self._motion_joint_pos_hist, -1, axis=0)
    self._motion_joint_pos_hist[-1] = self._robot_joint_pos
```

**问题**：
- `joint_pos`中只有手腕关节（索引23-28）有值，其他关节都是0
- 但encoder输入中使用了`motion_joint_positions_wrists_10frame_step1`（855行），这是正确的
- **然而**，decoder输入中使用了`self._robot_joint_pos_hist.flatten()`（949行），这包含了所有29个关节！

### 问题2：Decoder输入使用了错误的关节历史

**当前代码（action_provider_sonic.py:946-952）**：
```python
dec_obs = np.concatenate([
    latent.flatten(),                          # token_state: 64
    self._ang_vel_hist.flatten(),              # his_base_angular_velocity: 30
    self._robot_joint_pos_hist.flatten(),      # his_body_joint_positions: 290 ← 问题！
    self._robot_joint_vel_hist.flatten(),      # his_body_joint_velocities: 290 ← 问题！
    self._last_action_hist.flatten(),          # his_last_actions: 290
    self._grav_dir_hist.flatten(),             # his_gravity_dir: 30
])[np.newaxis]
```

**问题分析**：
- `_robot_joint_pos_hist`来自ZMQ的`joint_pos`，大部分是0
- `_robot_joint_vel_hist`来自ZMQ的`joint_vel`，全是0（pico_server发送的是`np.zeros((N, 29))`）
- 这导致decoder接收到的关节状态几乎全是0，无法正确推理！

**应该使用的数据**：
- 应该使用IsaacLab仿真中的实际机器人关节状态
- 在`_update_robot_hist_from_env()`函数（592-606行）中已经有这个逻辑

### 问题3：历史缓冲更新逻辑混乱

**当前有两个更新路径**：

1. **从ZMQ更新**（734-749行）：
   ```python
   if "joint_pos" in data:
       self._motion_joint_pos_hist = np.roll(...)
       self._motion_joint_pos_hist[-1] = self._robot_joint_pos  # 来自ZMQ
   ```

2. **从仿真环境更新**（592-606行）：
   ```python
   def _update_robot_hist_from_env(self):
       joint_pos_sonic = robot.joint_pos[0, self._sonic_idx].cpu().numpy()
       self._robot_joint_pos_hist = np.roll(...)
       self._robot_joint_pos_hist[-1] = joint_pos_sonic  # 来自仿真
   ```

**问题**：
- `_motion_joint_pos_hist`用于encoder输入（wrist部分）
- `_robot_joint_pos_hist`用于decoder输入（全身）
- 但两者的更新逻辑不一致，导致数据不同步

## 解决方案

### 方案A：修正decoder输入（推荐）

**修改action_provider_sonic.py:946-952**：

```python
# 在decoder推理前，先更新机器人状态历史
self._update_robot_hist_from_env()

# 使用仿真中的实际关节状态，而不是ZMQ的joint_pos
dec_obs = np.concatenate([
    latent.flatten(),                          # token_state: 64
    self._ang_vel_hist.flatten(),              # his_base_angular_velocity: 30
    self._robot_joint_pos_hist.flatten(),      # his_body_joint_positions: 290 ← 现在是正确的
    self._robot_joint_vel_hist.flatten(),      # his_body_joint_velocities: 290 ← 现在是正确的
    self._last_action_hist.flatten(),          # his_last_actions: 290
    self._grav_dir_hist.flatten(),             # his_gravity_dir: 30
])[np.newaxis]
```

**原理**：
- Encoder需要的是VR追踪数据（SMPL joints + wrist positions）
- Decoder需要的是机器人当前状态（joint positions + velocities）
- 两者的数据来源应该不同

### 方案B：完全移除ZMQ的joint_pos/joint_vel

**修改action_provider_sonic.py:733-749**：

```python
# 删除或注释掉这部分
# if "joint_pos" in data:
#     jp = data["joint_pos"].astype(np.float32)
#     ...
# if "joint_vel" in data:
#     jv = data["joint_vel"].astype(np.float32)
#     ...
```

**原理**：
- ZMQ的`joint_pos`/`joint_vel`本来就不完整（大部分是0）
- 应该完全依赖IsaacLab仿真中的机器人状态

## 验证方法

修改后，在IsaacLab日志中应该看到：

```
[SONIC][DECODER_INPUT] robot_joint_pos range: [-1.5, 1.5]  ← 不再是接近0
[SONIC][DECODER_INPUT] robot_joint_vel range: [-2.0, 2.0]  ← 不再是全0
```

如果decoder输入的关节状态有合理的数值范围，机器人应该能正确响应VR动作。

## 为什么之前没发现这个问题？

1. **诊断脚本的局限**：
   - `diagnose_sonic_dataflow.py`使用的是简化的测试数据
   - 没有模拟真实的IsaacLab环境
   - 无法检测到历史缓冲的数据来源问题

2. **日志不够详细**：
   - 现有日志主要关注encoder输入
   - decoder输入的日志较少
   - 没有打印`_robot_joint_pos_hist`的实际数值

3. **代码注释误导**：
   - 注释说"机器人关节状态（来自 ZMQ，用于 obs 构建）"
   - 但实际上ZMQ的joint_pos不完整，不应该用于decoder

## 下一步行动

1. **立即修改**：在`_run_gear_sonic()`函数中，decoder推理前调用`self._update_robot_hist_from_env()`

2. **添加调试日志**：
   ```python
   print(f"[DEBUG] _robot_joint_pos_hist range: [{self._robot_joint_pos_hist.min():.3f}, {self._robot_joint_pos_hist.max():.3f}]")
   print(f"[DEBUG] _robot_joint_vel_hist range: [{self._robot_joint_vel_hist.min():.3f}, {self._robot_joint_vel_hist.max():.3f}]")
   ```

3. **重新测试**：在VR中做动作，观察机器人是否正确响应

这很可能就是"有动作但不按照行为执行"的根本原因！
