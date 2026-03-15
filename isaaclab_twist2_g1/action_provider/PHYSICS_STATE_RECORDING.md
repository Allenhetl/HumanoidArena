# 物理状态记录说明

## 概述

在 `action_provider_wh_twist2.py` 中新增了物理状态的记录，用于更深入地分析机器人动力学行为。

## 新增记录的物理状态

### 1. Applied Torque (施加的力矩)

**字段名**: `robot_applied_torque_before_decimation`

**数据类型**: `numpy.ndarray [29]`

**含义**:
- PD控制器在**上一个控制周期**计算并施加到关节的力矩
- 这是物理引擎实际使用的力矩值
- 在当前控制周期的decimation loop之前采集

**特性**:
- ✅ **可以记录**: 从 `robot.data.applied_torque` 读取
- ❌ **不能直接写入**: 这是PD控制器的输出，不是输入
- 💡 **用途**:
  - 分析PD控制器的实际输出
  - 对比目标位置和实际力矩的关系
  - 诊断控制器饱和问题

**计算公式** (PD控制器):
```
applied_torque = Kp * (target_pos - current_pos) + Kd * (0 - current_vel)
```

### 2. Body Net Contact Forces (刚体净接触力)

**字段名**: `robot_body_net_contact_forces`

**数据类型**: `numpy.ndarray [num_bodies, 3]`

**含义**:
- 每个刚体受到的净接触力（世界坐标系）
- 包括所有接触点的合力
- 由物理引擎根据接触约束计算

**特性**:
- ✅ **可以记录**: 从 `robot.data.body_net_contact_force_w` 读取
- ❌ **不能直接写入**: 这是物理引擎计算的结果
- 💡 **用途**:
  - 分析机器人与环境的交互
  - 检测碰撞和接触状态
  - 验证物理仿真的合理性

**坐标系**: 世界坐标系 (world frame)

## 数据采集时机

所有物理状态都在**decimation loop之前**采集，即：

```
控制周期 N:
├─ 1. compute_observations()
│   └─ 读取: qpos, qvel, applied_torque (上一周期的结果)
├─ 2. run_policy()
│   └─ 推理: action
├─ 3. collect_recording_data()  ← 在这里记录所有状态
│   ├─ qpos_before_decimation
│   ├─ qvel_before_decimation
│   ├─ applied_torque_before_decimation (上一周期的)
│   └─ body_net_contact_forces
└─ 4. decimation loop (10步)
    └─ PD控制器作用 → 产生新的 applied_torque
```

## Replay时的处理

### 可以写入的状态

在 `action_provider_wh_twist2_replay.py` 中，以下状态可以写入：

```python
# ✅ 可以写入
robot.write_root_state_to_sim(root_state)  # 位置、姿态、速度
robot.write_joint_state_to_sim(position, velocity)  # 关节位置、速度
```

### 不能写入的状态

以下状态**不能直接写入**，但可以用于分析：

```python
# ❌ 不能写入
# applied_torque - 由PD控制器计算，不是输入
# body_net_contact_forces - 由物理引擎计算，不是输入
```

### Replay策略

1. **设置初始状态**: 写入位置和速度
2. **应用action**: 设置PD目标位置
3. **让物理引擎计算**:
   - PD控制器重新计算 `applied_torque`
   - 物理引擎重新计算 `contact_forces`

## 数据分析用途

### 1. 对比分析

```python
# 对比recording和replay的力矩差异
recording_torque = data['robot_applied_torque_before_decimation']
replay_torque = robot.data.applied_torque[0, indices].cpu().numpy()
torque_diff = np.abs(recording_torque - replay_torque)
```

### 2. 接触检测

```python
# 检测哪些刚体有接触
contact_forces = data['robot_body_net_contact_forces']
contact_threshold = 1.0  # N
in_contact = np.linalg.norm(contact_forces, axis=1) > contact_threshold
```

### 3. 控制器饱和分析

```python
# 检查力矩是否达到限制
torque_limits = robot_config['effort_limits']  # 从URDF读取
saturation = np.abs(applied_torque) > (0.9 * torque_limits)
```

## 数据存储

所有数据保存在 `.npz` 文件中：

```python
data = np.load('episode_xxx.npz', allow_pickle=True)

# 访问新增的物理状态
applied_torque = data['robot_applied_torque_before_decimation']  # [num_frames, 29]
contact_forces = data['robot_body_net_contact_forces']  # [num_frames, num_bodies, 3]
```

## 注意事项

1. **时序对齐**: `applied_torque_before_decimation` 是上一个周期的输出，不是当前周期的
2. **坐标系**: 接触力使用世界坐标系，需要注意坐标变换
3. **数据大小**: 接触力数据较大（num_bodies × 3），会增加存储空间
4. **异常处理**: 如果读取失败，字段值为 `None`

## 未来扩展

可以考虑记录的其他物理状态：

- `body_acc_w`: 刚体加速度
- `joint_acc`: 关节加速度
- `contact_sensor_data`: 接触传感器数据（如果有）
- `external_forces`: 外部施加的力（如果有）

但需要权衡存储空间和分析价值。
