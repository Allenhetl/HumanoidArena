# 录制 vs Replay 执行流程对比

## 录制脚本 (action_provider_wh_twist2.py)

### 初始化阶段
```python
def __init__(self, env, args_cli):
    # 没有调用任何 write_* 方法
    # 机器人保持spawn默认状态
```

### 每帧执行流程 (get_action)
```python
# Frame N 开始时的状态：
# - joint_pos: 上一帧decimation后的实际位置
# - root_state: 上一帧decimation后的实际状态

# 1. Policy推理
action_data, obs_buf = self.run_policy()
target_29 = raw_action * 0.5 + self.twist2_default_pos

# 2. 收集录制数据（在decimation之前）
recording_data = self.collect_recording_data(obs_buf, target_29)
# 读取：
# - qpos_before_decimation = self.joint_pos[0, idx]  # 当前帧开始时
# - root_state = self.env.scene["robot"].data.root_state_w  # 当前帧开始时
# - twist2_inference_qpos = target_29  # 本帧的目标

# 3. 填充full_action
full_action.copy_(default_joint_pos)
full_action.index_copy_(0, twist2_action_indices, target_29)

# 4. Decimation循环（10步）
for i in range(10):
    self.env.scene["robot"].set_joint_position_target(full_action)
    self.env.scene.write_data_to_sim()
    self.env.sim.step(render=...)
    self.env.scene.update(dt=self.env.physics_dt)

# Frame N 结束时：
# - joint_pos: 更新为decimation后的新位置
# - root_state: 更新为decimation后的新状态
```

## Replay脚本 (action_provider_wh_twist2_replay.py)

### 初始化阶段
```python
def __init__(self, env, args_cli):
    # 加载录制数据
    self.replay_data_qpos_actual = data['robot_qpos_before_decimation']
    self.replay_data_qvel = data['robot_qvel_before_decimation']
    self.replay_data_root_pos = data['robot_root_position']
    self.replay_data_root_quat = data['robot_root_orientation']
```

### Frame 0 执行流程
```python
if self.current_frame == 0:
    # 设置初始joint positions和velocities
    initial_qpos = self.replay_data_qpos_actual[0]
    initial_qvel = self.replay_data_qvel[0]

    full_initial_pos[0, twist2_action_indices] = initial_qpos
    full_initial_vel[0, twist2_action_indices] = initial_qvel

    # ⚠️ 关键调用
    self.env.scene["robot"].write_joint_state_to_sim(
        position=full_initial_pos,
        velocity=full_initial_vel
    )
    self.env.scene.write_data_to_sim()
```

### 每帧执行流程 (get_action)
```python
# Frame N 开始时的状态：
# - Frame 0: 通过write_joint_state_to_sim设置的状态
# - Frame > 0: 上一帧decimation后的状态

# 1. 获取target_29
if replay_mode == "inference":
    # 使用录制的observation重新推理
    obs_tensor = torch.from_numpy(self.replay_data_obs[current_frame])
    action_data = self.policy.run(None, {input_name: obs_tensor})
    target_29 = raw_action * 0.5 + self.twist2_default_pos
else:
    # 直接使用录制的target
    target_29 = torch.from_numpy(self.replay_data_qpos[current_frame])

# 2. 填充full_action
full_action.copy_(default_joint_pos)
full_action.index_copy_(0, twist2_action_indices, target_29)

# 3. Decimation循环（10步）- 与录制完全相同
for i in range(10):
    self.env.scene["robot"].set_joint_position_target(full_action)
    self.env.scene.write_data_to_sim()
    self.env.sim.step(render=...)
    self.env.scene.update(dt=self.env.physics_dt)

# Frame N 结束时：
# - joint_pos: 更新为decimation后的新位置
# - root_state: 更新为decimation后的新状态
```

## 关键差异

### 1. Frame 0的初始化
| 方面 | 录制脚本 | Replay脚本 |
|------|---------|-----------|
| Joint positions | Spawn默认值 | 通过write_joint_state_to_sim设置为录制的qpos_actual[0] |
| Joint velocities | 0 | 通过write_joint_state_to_sim设置为录制的qvel[0] |
| Root position | Spawn默认值 | **不设置**（当前版本） |
| Root velocity | 0 | **不设置**（当前版本） |

### 2. write_joint_state_to_sim的影响

**可能的副作用：**
1. 重置物理引擎的内部状态
2. 清除或重置约束
3. 影响root的物理属性（质量、惯性等）
4. 重置接触状态

### 3. 问题现象

**用户报告：**
- "初始化有设置位置，但执行过程中根节点移动到后下方然后固定住了"
- "感觉似乎启动过后一段时间后根节点就不能动了"

**可能原因：**
1. `write_joint_state_to_sim`改变了root的物理状态
2. 缺少root velocity的初始化
3. 缺少某些同步调用
4. 物理引擎的约束或阻尼设置

## 下一步调查方向

1. **测试不同的初始化顺序**
   - 先设置root再设置joints
   - 先设置joints再设置root
   - 分别调用write_data_to_sim

2. **检查是否需要额外的同步**
   - 调用sim.step()
   - 调用scene.update()
   - 重置物理引擎状态

3. **对比物理引擎状态**
   - 录制时的root物理属性
   - Replay时的root物理属性
   - 检查约束、阻尼等参数

4. **测试最小化差异**
   - 只设置joint positions，不设置velocities
   - 不调用write_data_to_sim
   - 使用set_joint_position_target代替write_joint_state_to_sim
