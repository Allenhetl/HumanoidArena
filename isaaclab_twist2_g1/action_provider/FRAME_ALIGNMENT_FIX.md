# Frame Alignment Fix - 状态与动作对齐修复

## 问题分析

### 问题1：状态-动作时序错位
**之前的错误逻辑：**
```
Frame 0: 设置状态 → 执行Frame 0动作 ❌
```
- Frame 0的状态是**执行Frame 0动作后的结果**
- 设置Frame 0状态后又执行Frame 0动作 → 状态和动作不匹配

**正确的逻辑：**
```
Frame 0: 仅用于初始化（设置状态）
Frame 1: 执行Frame 1动作 ✅
Frame 2: 执行Frame 2动作 ✅
```

### 问题2：使用了错误的关节位置数据
**关键区别：**
- `qpos` (robot_twist2_inference_qpos) = PD控制器的**目标位置**（ONNX输出）
- `qpos_actual` (robot_qpos_before_decimation) = 物理仿真后的**实际位置**

**错误：** 初始化时使用 `qpos[0]`（目标位置）
**正确：** 初始化时使用 `qpos_actual[0]`（实际位置）

**为什么必须用实际位置：**
- PD控制器有跟踪误差，目标位置 ≠ 实际位置
- Frame 0的实际状态是录制时的真实状态
- 如果用目标位置初始化，会产生巨大的初始误差

## 修改内容

### 1. 初始化时从Frame 1开始
```python
# action_provider_wh_twist2_replay.py:69
self.current_frame = 1  # 从Frame 1开始（Frame 0用于初始化）
```

### 2. 使用实际位置初始化（关键修复！）
```python
# action_provider_wh_twist2_replay.py:337
if self.replay_data_qpos_actual is not None:
    initial_qpos = self.replay_data_qpos_actual[0]  # 使用实际位置
```

**对比：**
```python
# ❌ 错误：使用目标位置
initial_qpos = self.replay_data_qpos[0]  # ONNX输出的目标

# ✅ 正确：使用实际位置
initial_qpos = self.replay_data_qpos_actual[0]  # 物理仿真后的实际状态
```

### 3. 更新初始化方法说明
```python
def set_initial_state_from_recording(self):
    """使用Frame 0的实际状态初始化，replay从Frame 1开始

    确保状态-动作对齐：Frame 0实际状态 + Frame 1目标动作
    """
```

### 4. Loop时回到Frame 1
```python
if self.replay_loop:
    self.current_frame = 1  # 循环回到Frame 1（Frame 0是初始化）
```

## 数据完备性验证

✅ **逻辑完备**
- Frame 0: 有完整实际状态数据（root_pos, root_vel, qpos_actual, qvel）→ 用于初始化
- Frame 1~N: 有目标动作数据（qpos）→ 用于replay
- 状态和动作完美对齐

✅ **时序正确**
```
录制时：
  Frame 0实际状态 → 执行qpos[1] → Frame 1实际状态 → 执行qpos[2] → Frame 2实际状态

Replay时：
  设置Frame 0实际状态 → 执行qpos[1] → 对比Frame 1状态 ✅
```

✅ **数据类型正确**
```
初始化：qpos_actual[0] (实际位置) + qvel[0] (实际速度)
执行：  qpos[1] (目标位置，PD控制器输入)
```

## 预期效果

- **Frame 1的初始误差应该接近0**（因为使用了实际状态初始化）
- 关节位置误差：从 0.305 rad → <0.05 rad
- 关节速度误差：从 18.21 rad/s → <2.0 rad/s
- Root速度误差：从 0.528 m/s → <0.1 m/s
- 误差累积应该显著减少
