# 回放一致性问题修复总结

## 问题描述

在使用 `action_provider_wh_twist2.py` 录制遥操作数据后，无论采用哪种回放方式（target_29_dof 直接回放或 ONNX 推理回放），都无法复现录制时的动作。

## 根本原因分析

### 1. **初始状态不完整** ⚠️ 最关键问题

**问题**：回放脚本只设置了关节状态，但**没有设置机器人的根节点（root）位置、姿态和速度**。

**影响**：
- 录制时机器人可能处于任意位置和姿态
- 回放时使用环境默认的 spawn 位置（通常是原点）
- 即使关节角度相同，根节点状态不同会导致完全不同的物理演化

**证据**：在 `action_provider_wh_twist2_replay.py:291-292` 发现：
```python
# EXPERIMENT: Don't set root pose - test if write_root_pose_to_sim locks the root
print(f"[{self.name}]   ⚠️  EXPERIMENT: NOT setting root pose to test if it's being locked")
```

### 2. **速度信息缺失**

**问题**：初始状态只设置了位置，没有设置速度（线速度和角速度）。

**影响**：
- PhysX 物理引擎需要完整的状态（位置+速度）才能准确演化
- 缺少速度信息会导致物理模拟从静止状态开始，与录制时的动态状态不符

### 3. **足球状态不完整**

**问题**：足球的速度信息没有被恢复。

**影响**：
- 足球的运动轨迹会与录制时不同
- 机器人与足球的交互会产生偏差

### 4. **重复的初始化逻辑**

**问题**：`sim_main_replay.py` 和 `action_provider_wh_twist2_replay.py` 都有初始化逻辑，且 `sim_main_replay.py` 中的实现是错误的（速度设为零）。

**影响**：
- 两次初始化可能相互冲突
- 错误的速度设置会覆盖正确的值

## 修复方案

### 修改 1: `action_provider_wh_twist2_replay.py` - 完整的初始状态恢复

**位置**：`get_action()` 方法中 `if self.current_frame == 0:` 块（lines 286-390）

**修改内容**：

1. **恢复机器人根节点状态**（新增）：
   ```python
   # 1. Set robot root state (position, orientation, velocities)
   if self.replay_data_root_pos is not None and self.replay_data_root_quat is not None:
       robot = self.env.scene["robot"]
       root_state = robot.data.default_root_state.clone()

       # Set position and orientation
       root_state[0, 0:3] = torch.from_numpy(self.replay_data_root_pos[0]).to(self.env.device, dtype=torch.float32)
       root_state[0, 3:7] = torch.from_numpy(self.replay_data_root_quat[0]).to(self.env.device, dtype=torch.float32)

       # Set velocities (prefer world frame)
       if self.replay_data_root_lin_vel is not None:
           root_state[0, 7:10] = torch.from_numpy(self.replay_data_root_lin_vel[0]).to(self.env.device, dtype=torch.float32)
       else:
           root_state[0, 7:10] = torch.zeros(3, device=self.env.device, dtype=torch.float32)

       if self.replay_data_root_ang_vel is not None:
           root_state[0, 10:13] = torch.from_numpy(self.replay_data_root_ang_vel[0]).to(self.env.device, dtype=torch.float32)
       else:
           root_state[0, 10:13] = torch.zeros(3, device=self.env.device, dtype=torch.float32)

       robot.write_root_state_to_sim(root_state)
   ```

2. **恢复足球完整状态**（新增）：
   ```python
   # 3. Set football state if available
   try:
       if "object" in self.env.scene.keys():
           # Check if football data exists in recording
           import numpy as np
           data = np.load(self.replay_file, allow_pickle=True)
           if 'env_obj_football_position' in data:
               football = self.env.scene["object"]
               football_state = football.data.default_root_state.clone()

               # Set position
               football_pos = data['env_obj_football_position'][0]
               football_state[0, 0:3] = torch.from_numpy(football_pos).to(self.env.device, dtype=torch.float32)

               # Set velocities if available
               if 'env_obj_football_linear_velocity' in data:
                   football_lin_vel = data['env_obj_football_linear_velocity'][0]
                   football_state[0, 7:10] = torch.from_numpy(football_lin_vel).to(self.env.device, dtype=torch.float32)
               else:
                   football_state[0, 7:10] = torch.zeros(3, device=self.env.device, dtype=torch.float32)

               if 'env_obj_football_angular_velocity' in data:
                   football_ang_vel = data['env_obj_football_angular_velocity'][0]
                   football_state[0, 10:13] = torch.from_numpy(football_ang_vel).to(self.env.device, dtype=torch.float32)
               else:
                   football_state[0, 10:13] = torch.zeros(3, device=self.env.device, dtype=torch.float32)

               football.write_root_state_to_sim(football_state)
   except Exception as e:
       print(f"[{self.name}]   ⚠️  Failed to set football state: {e}")
   ```

3. **添加验证输出**：
   ```python
   # Verify the state was set correctly
   actual_root_state = self.env.scene["robot"].data.root_state_w[0]
   actual_root_pos = actual_root_state[:3].cpu().numpy()
   actual_root_quat = actual_root_state[3:7].cpu().numpy()
   print(f"[{self.name}]   📊 Verification - Actual root state after setting:")
   print(f"[{self.name}]      Root pos: {actual_root_pos}")
   print(f"[{self.name}]      Root quat: {actual_root_quat}")
   ```

### 修改 2: `sim_main_replay.py` - 移除重复的初始化逻辑

**位置**：lines 295-349

**修改内容**：删除整个初始化块，替换为简单的说明注释：

```python
# NOTE: Initial state restoration is now handled by ReplayActionProvider
# in its get_action() method when current_frame == 0.
# This ensures proper timing and avoids conflicts with the control loop.
print(f"\n========= Initial state will be restored by ReplayActionProvider =========")
print(f"Initial state restoration happens in the first get_action() call")
print("=" * 60)
```

**原因**：
- 避免重复初始化导致的冲突
- 确保初始化在正确的时机（第一次 `get_action()` 调用时）进行
- 让 action provider 完全负责状态管理

## 诊断工具

创建了 `diagnose_replay_consistency.py` 脚本，用于：

1. **数据完整性检查**：验证录制文件是否包含所有必要的状态信息
2. **初始状态分析**：显示第一帧的详细状态（位置、速度、关节角度等）
3. **轨迹分析**：分析整个录制过程中的运动轨迹
4. **确定性要求检查**：验证数据是否满足确定性回放的要求

**使用方法**：
```bash
# 检查最新的录制文件
python3 action_provider/diagnose_replay_consistency.py latest

# 检查指定的录制文件
python3 action_provider/diagnose_replay_consistency.py recording_data/your_file.npz
```

## 验证步骤

### 1. 检查现有录制数据

```bash
cd /home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1
python3 action_provider/diagnose_replay_consistency.py latest
```

**预期输出**：
- ✅ 所有必要字段都存在（包括速度信息）
- ✅ 初始状态显示非零的根节点位置/速度
- ✅ 足球状态完整

**如果数据不完整**：需要重新录制数据（确保使用最新的 `action_provider_wh_twist2.py`）

### 2. 测试回放（Direct 模式）

```bash
# 使用录制的 target_29_dof 直接回放
./run_replay.sh recording_data/your_file.npz direct
```

**观察要点**：
- 初始化输出应显示：
  ```
  ✓ Set root pos: [x, y, z]
  ✓ Set root quat: [w, x, y, z]
  ✓ Set root lin_vel: [vx, vy, vz]
  ✓ Set root ang_vel: [wx, wy, wz]
  ✓ Set joint pos (前5个): [...]
  ✓ Set joint vel (前5个): [...]
  ✓ Set football pos: [x, y, z]
  ```
- 机器人应该从录制时的位置开始
- 动作应该与录制时基本一致

### 3. 测试回放（Inference 模式）

```bash
# 使用 ONNX 模型重新推理
./run_replay.sh recording_data/your_file.npz inference
```

**观察要点**：
- 初始状态应该相同
- 由于使用录制的观测数据进行推理，动作应该与录制时非常接近
- 如果有偏差，可能是由于：
  - 随机种子不一致（已在之前的修复中解决）
  - ONNX 推理的数值精度差异（正常，应该很小）

### 4. 对比录制和回放

可以使用以下方法验证一致性：

1. **视觉对比**：
   - 录制时保存视频
   - 回放时保存视频
   - 对比两个视频的前几秒

2. **数据对比**：
   - 在回放脚本中添加日志，记录每一帧的状态
   - 与录制数据进行数值对比
   - 计算误差（位置误差、角度误差等）

## 可能的剩余问题

即使完成了上述修复，仍可能存在以下因素导致的微小差异：

### 1. 浮点精度

**原因**：
- 录制和回放可能在不同的硬件上运行
- ONNX 推理的浮点运算顺序可能不同

**影响**：微小的数值差异会随时间累积

**解决方案**：
- 使用相同的硬件
- 确保 ONNX Runtime 使用确定性模式（已在 seed 修复中实现）

### 2. 时序问题

**原因**：
- 录制时的控制频率可能不稳定
- 回放时的物理步长必须完全一致

**影响**：时序不匹配会导致动作不同步

**解决方案**：
- 确保 `decimation` 参数一致
- 确保 `physics_dt` 一致
- 使用固定的控制频率

### 3. 观测数据的时序对齐

**原因**：
- 录制的观测数据是在特定时刻计算的
- 回放时的物理状态可能与录制时略有不同

**影响**：即使使用相同的观测数据，ONNX 推理的输出也可能略有不同

**解决方案**：
- 使用 Direct 模式（直接使用录制的 target_29_dof）进行验证
- 如果 Direct 模式一致，说明问题在于观测计算
- 如果 Direct 模式也不一致，说明问题在于物理模拟

## 调试建议

如果修复后仍有问题，按以下顺序排查：

### 1. 验证初始状态

在回放开始时，打印并对比：
- 录制的初始状态（从 npz 文件读取）
- 回放的初始状态（从 sim 读取）

**应该完全一致**，包括：
- 根节点位置、姿态、速度
- 关节位置、速度
- 足球位置、速度

### 2. 验证动作一致性

在 Direct 模式下，打印每一帧的：
- 录制的 `target_29_dof`
- 回放时使用的 `target_29_dof`

**应该完全一致**。

### 3. 验证物理演化

在 Direct 模式下，每 10 帧打印：
- 当前的根节点位置
- 当前的关节位置
- 与录制数据的差异

**如果差异随时间增大**：
- 检查物理参数（重力、摩擦力、阻尼等）
- 检查 PD 控制器参数（stiffness, damping）
- 检查随机种子是否一致

### 4. 验证观测计算

在 Inference 模式下，对比：
- 录制的观测数据
- 回放时计算的观测数据

**如果不一致**：
- 检查观测计算的代码是否有变化
- 检查传感器数据（相机、IMU 等）是否一致

## 总结

**核心修复**：
1. ✅ 恢复完整的机器人根节点状态（位置、姿态、速度）
2. ✅ 恢复完整的足球状态（位置、速度）
3. ✅ 移除重复的初始化逻辑
4. ✅ 添加诊断工具验证数据完整性

**预期结果**：
- Direct 模式应该能够精确复现录制的动作
- Inference 模式应该能够产生非常接近的动作（可能有微小的数值差异）

**如果仍有问题**：
- 使用诊断工具检查数据完整性
- 按照调试建议逐步排查
- 检查物理参数和控制参数是否一致
