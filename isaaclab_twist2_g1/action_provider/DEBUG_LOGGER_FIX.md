# Debug Logger Fix - 修复对比数据错误

## 问题发现

Debug日志显示巨大误差，但实际replay效果很好！

### 错误的对比
```python
# ❌ 之前的代码（第604行）
recorded['joint_pos'] = torch.from_numpy(self.replay_data_qpos[frame_idx])
```

**问题：** 对比 simulated(实际位置) vs qpos(目标位置)

### 数据分析

**Frame 1 实际数据：**
```
qpos_actual[1] (录制时的实际位置): [0.136, -0.083, 0.056, -0.025, -0.051]
qpos[1] (录制时的目标位置):       [-0.051, 0.128, -0.053, -0.062, 0.081]
simulated[1] (replay仿真结果):    [0.118, -0.086, 0.067, -0.018, -0.051]
```

**误差对比：**
- ❌ simulated vs qpos (debug日志显示): **0.329 rad**
- ✅ simulated vs qpos_actual (真实误差): **0.022 rad**

**结论：** Replay效果实际上非常好！误差仅0.022 rad，但debug日志显示0.329 rad是因为对比了错误的数据。

## 修复方案

### 使用实际位置对比
```python
# ✅ 修复后的代码
if self.replay_data_qpos_actual is not None:
    recorded['joint_pos'] = torch.from_numpy(self.replay_data_qpos_actual[frame_idx])
elif self.replay_data_qpos is not None:
    # Fallback: 如果没有actual数据，使用target（会显示更大误差）
    recorded['joint_pos'] = torch.from_numpy(self.replay_data_qpos[frame_idx])
    print("⚠️ WARNING: Using target positions for comparison")
```

## 为什么要对比实际位置

### PD控制器的工作原理
```
录制时：
  qpos[t] (目标) → PD控制器 → qpos_actual[t] (实际)

Replay时：
  qpos[t] (目标) → PD控制器 → simulated[t] (实际)
```

### 正确的对比
- ✅ **simulated vs qpos_actual**: 对比两次运行的实际结果
- ❌ **simulated vs qpos**: 对比实际结果和目标（无意义）

### PD跟踪误差
- PD控制器有跟踪误差，实际位置 ≠ 目标位置
- 目标和实际可能相差很大（0.1-0.3 rad）
- 对比目标会显示虚假的巨大误差

## 预期效果

修复后，debug日志应该显示：
- Frame 1 关节位置误差：从 0.733 rad → **0.022 rad** ✅
- Frame 1 关节速度误差：从 8.99 rad/s → **<2.0 rad/s** ✅
- 整体误差曲线应该平缓，不会有虚假的巨大误差

## 测试验证

使用测试数据重新运行replay：
```bash
# 测试数据
/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/Isaac-Move-Football-G129-Dex3-Wholebody_1773326197574686.npz

# Conda环境
unitree_sim_env_isaaclab5_0
```

预期Frame 1误差：
- Root位置: <0.01m ✅
- Root姿态: <0.01 ✅
- 关节位置: <0.05 rad ✅
- 关节速度: <2.0 rad/s ✅
