# 初始状态验证日志说明

## 功能

在replay初始化后，自动读取robot的实际状态并写入txt日志文件，用于验证初始化是否正确。

## 日志文件位置

```
./replay_debug_logs/<recording_name>_initial_state_verification.txt
```

例如：
```
./replay_debug_logs/Isaac-Move-Football-G129-Dex3-Wholebody_1773326197574686_initial_state_verification.txt
```

## 日志内容

### 1. 预期状态（Frame 0录制数据）
- Root Position
- Root Quaternion (w,x,y,z)
- Root Linear Velocity
- Root Angular Velocity
- Joint Positions (29个关节)
- Joint Velocities (29个关节)

### 2. 实际状态（初始化后读取）
- 从物理引擎读取的实际状态
- 包含所有与预期状态相同的字段

### 3. 误差分析
- Root Position Error (L2 norm)
- Root Quaternion Error (L2 norm)
- Root Linear Velocity Error (L2 norm)
- Root Angular Velocity Error (L2 norm)
- Joint Position Error (L2 norm 和 Max)
- Joint Velocity Error (L2 norm 和 Max)

### 4. 逐关节对比（前5个关节）
- 每个关节的预期值、实际值、误差

## 如何使用

### 1. 运行replay
```bash
python sim_main_replay.py \
    --task Isaac-Move-Football-G129-Dex3-Wholebody \
    --replay_file recording_data/xxx.npz \
    --seed 42
```

### 2. 查看日志
```bash
cat replay_debug_logs/xxx_initial_state_verification.txt
```

### 3. 分析结果

**预期结果（初始化正确）：**
- Root Position Error: <0.001 m
- Root Quaternion Error: <0.001
- Root Velocity Error: <0.01 m/s
- Joint Position Error: <0.001 rad
- Joint Velocity Error: <0.1 rad/s

**如果误差很大：**
- Root Position Error > 0.01 m → 位置没有正确设置
- Joint Position Error > 0.1 rad → 关节位置没有正确设置
- Velocity Error > 1.0 → 速度没有正确设置

## 示例日志

```
================================================================================
Initial State Verification - 初始化后的实际状态
生成时间: 2026-03-13 12:00:00
================================================================================

【预期状态 - Frame 0录制数据】
--------------------------------------------------------------------------------
Root Position:    [-1.9035457 -5.0124316  0.7892066]
Root Quaternion:  [ 0.49719247  0.01415497 -0.04191624  0.8665116 ]
Root Lin Vel:     [-0.19346255  0.10749328  0.09504685]
Root Ang Vel:     [ 0.12004934 -0.5462194  -0.69921905]
Joint Pos (前5个): [ 0.07924579 -0.11440853  0.02214489  0.0413378  -0.05671784]
Joint Vel (前5个): [ 0.48820585  0.59121305  0.43602392 -0.7166607  -0.08921745]

【实际状态 - 初始化后读取】
--------------------------------------------------------------------------------
Root Position:    [-1.9035457 -5.0124316  0.7892066]
Root Quaternion:  [ 0.49719247  0.01415497 -0.04191624  0.8665116 ]
Root Lin Vel:     [-0.19346255  0.10749328  0.09504685]
Root Ang Vel:     [ 0.12004934 -0.5462194  -0.69921905]
Joint Pos (前5个): [ 0.07924579 -0.11440853  0.02214489  0.0413378  -0.05671784]
Joint Vel (前5个): [ 0.48820585  0.59121305  0.43602392 -0.7166607  -0.08921745]

【误差分析】
--------------------------------------------------------------------------------
Root Position Error (L2):    0.000000 m
Root Quaternion Error (L2):  0.000000
Root Lin Vel Error (L2):     0.000000 m/s
Root Ang Vel Error (L2):     0.000000 rad/s
Joint Position Error (L2):   0.000000 rad
Joint Position Error (Max):  0.000000 rad (joint 0)
Joint Velocity Error (L2):   0.000000 rad/s
Joint Velocity Error (Max):  0.000000 rad/s (joint 0)

【逐关节对比 (前5个关节)】
--------------------------------------------------------------------------------
Joint 0: 预期= 0.07925, 实际= 0.07925, 误差= 0.00000 rad
Joint 1: 预期=-0.11441, 实际=-0.11441, 误差= 0.00000 rad
Joint 2: 预期= 0.02214, 实际= 0.02214, 误差= 0.00000 rad
Joint 3: 预期= 0.04134, 实际= 0.04134, 误差= 0.00000 rad
Joint 4: 预期=-0.05672, 实际=-0.05672, 误差= 0.00000 rad

================================================================================
```

## 故障排查

### 问题1: 日志文件未生成

**可能原因：**
- 初始化失败
- 没有权限写入文件

**解决方案：**
- 检查控制台是否有错误信息
- 确保 `./replay_debug_logs` 目录可写

### 问题2: 误差很大

**可能原因：**
1. `write_data_to_sim()` 没有立即生效
2. 需要运行一个physics step才能生效
3. 数据类型转换错误

**解决方案：**
- 检查是否调用了 `write_data_to_sim()`
- 尝试在设置状态后运行一个physics step
- 检查数据类型和设备（CPU vs GPU）

### 问题3: 某些关节误差特别大

**可能原因：**
- 关节索引映射错误
- 某些关节的限制导致无法设置到目标值

**解决方案：**
- 检查 `twist2_action_indices` 映射
- 检查关节限制（position limits）

## 与Frame 1误差的关系

如果初始状态设置正确（误差接近0），但Frame 1仍有大误差，说明问题在于：
1. 执行的动作不正确
2. PD控制器参数不同
3. Decimation或physics dt不同
4. 物理引擎的随机性

如果初始状态就有大误差，说明问题在于：
1. 初始化代码有bug
2. `write_data_to_sim()` 没有生效
3. 数据加载错误
