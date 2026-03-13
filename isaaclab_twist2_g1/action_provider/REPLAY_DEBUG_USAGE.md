# Replay Debug Logger 使用指南

## 功能说明

`ReplayDebugLogger` 用于在replay过程中对比录制数据和仿真状态，帮助定位replay失败的原因。

## 自动启用

Debug logging **默认启用**，无需额外配置。运行replay时会自动生成以下文件：

```
./replay_debug_logs/
├── <recording_name>.txt          # 详细的逐帧对比日志（人类可读）
├── <recording_name>.json         # 结构化数据（用于分析脚本）
└── <recording_name>_summary.txt  # 摘要报告（推荐先看这个）
```

**文件用途**：
- `.txt` - 用grep查找特定帧的详细信息
- `.json` - 用于Python分析脚本（`analyze_replay_errors.py`）
- `_summary.txt` - 快速查看最大误差和建议

## 运行replay

```bash
# 正常运行replay，会自动生成debug日志
bash run_replay.sh

# 或者直接运行
python sim_main_replay.py \
    --replay_file ./recording_data/recording_20250313_120000.npz \
    --replay_mode direct
```

## 禁用debug logging（如果需要）

如果不想生成日志（例如性能测试），可以添加参数：

```bash
python sim_main_replay.py \
    --replay_file ./recording_data/recording_20250313_120000.npz \
    --replay_mode direct \
    --enable_replay_debug_log False
```

## 自定义日志目录

```bash
python sim_main_replay.py \
    --replay_file ./recording_data/recording_20250313_120000.npz \
    --replay_mode direct \
    --replay_debug_log_dir ./my_debug_logs
```

## 日志文件说明

### 1. 详细日志 (.txt)

逐帧记录录制数据和仿真状态的对比：

```
====================================================================================================
Frame 90
====================================================================================================

Root Position:
  Recorded:  [1.234567, -0.123456, 0.987654] m
  Simulated: [1.234500, -0.123400, 0.987600] m
  Error (L2): 0.000123 m
  Error (Max): 0.000067 m

Root Quaternion (w,x,y,z):
  Recorded:  [0.999999, 0.000123, -0.000456, 0.000789]
  Simulated: [0.999998, 0.000120, -0.000450, 0.000785]
  Error (L2): 0.000012
  Error (Max): 0.000006

Root Linear Velocity:
  Recorded:  [0.123456, -0.012345, 0.001234] m/s
  Simulated: [0.123400, -0.012300, 0.001200] m/s
  Error (L2): 0.000089 m/s
  Error (Max): 0.000056 m/s

Root Angular Velocity:
  Recorded:  [0.012345, -0.001234, 0.000123] rad/s
  Simulated: [0.012300, -0.001200, 0.000120] rad/s
  Error (L2): 0.000067 rad/s
  Error (Max): 0.000045 rad/s

Joint Positions (29 DOFs):
  Error (L2): 0.012345 rad
  Error (Max): 0.005678 rad (joint 15)
  Error (Mean): 0.000234 rad
  First 5 joints recorded:  [0.123456, -0.234567, 0.345678, -0.456789, 0.567890]
  First 5 joints simulated: [0.123400, -0.234500, 0.345600, -0.456700, 0.567800]
  Max error joint [15]: rec=0.123456, sim=0.117778

Joint Velocities (29 DOFs):
  Error (L2): 0.023456 rad/s
  Error (Max): 0.012345 rad/s (joint 8)
  Error (Mean): 0.000567 rad/s
```

### 2. 摘要报告 (_summary.txt)

统计整个replay过程中的最大误差：

```
====================================================================================================
Replay Debug Summary - 最大误差统计
生成时间: 2025-03-13 14:30:00
====================================================================================================

总帧数: 150

最大误差 (L2范数):
----------------------------------------------------------------------------------------------------
  root_pos            : 0.123456  (Frame 92)
  root_quat           : 0.012345  (Frame 95)
  root_lin_vel        : 0.234567  (Frame 90)
  root_ang_vel        : 0.034567  (Frame 88)
  joint_pos           : 0.156789  (Frame 93)
  joint_vel           : 0.345678  (Frame 91)

====================================================================================================
建议:
----------------------------------------------------------------------------------------------------
⚠️  Root位置误差较大 (0.123m)，在Frame 92达到峰值
   建议检查: 1) 初始状态设置 2) 物理参数 3) 接触力模型

⚠️  Root线速度误差较大 (0.235m/s)，在Frame 90达到峰值
   建议检查: 1) 摩擦系数 2) 地面接触 3) 质量分布

⚠️  关节位置误差较大 (0.157rad)，在Frame 93达到峰值
   建议检查: 1) PD控制器参数 2) 关节限位 3) 动作目标设置
```

### 3. JSON数据 (.json)

结构化数据，可用于Python分析和绘图：

```json
{
  "metadata": {
    "total_frames": 150,
    "timestamp": "2025-03-13T14:30:00"
  },
  "max_errors": {
    "root_pos": 0.123456,
    "root_quat": 0.012345,
    "root_lin_vel": 0.234567,
    "root_ang_vel": 0.034567,
    "joint_pos": 0.156789,
    "joint_vel": 0.345678
  },
  "max_error_frames": {
    "root_pos": 92,
    "root_quat": 95,
    "root_lin_vel": 90,
    "root_ang_vel": 88,
    "joint_pos": 93,
    "joint_vel": 91
  },
  "frames": [
    {
      "frame": 0,
      "recorded": {
        "root_pos": [1.0, 0.0, 0.5],
        "root_quat": [1.0, 0.0, 0.0, 0.0],
        ...
      },
      "simulated": {
        "root_pos": [1.0, 0.0, 0.5],
        "root_quat": [1.0, 0.0, 0.0, 0.0],
        ...
      },
      "errors": {
        "root_pos_l2": 0.0,
        "root_pos_max": 0.0,
        ...
      }
    },
    ...
  ]
}
```

## 分析误差的Python脚本示例

```python
import json
import matplotlib.pyplot as plt
import numpy as np

# 加载JSON数据（注意：使用.json文件，不是.txt文件）
with open('./replay_debug_logs/recording_20250313_120000.json', 'r') as f:
    data = json.load(f)

# 提取误差数据
frames = [f['frame'] for f in data['frames']]
root_pos_errors = [f['errors']['root_pos_l2'] for f in data['frames']]
joint_pos_errors = [f['errors']['joint_pos_l2'] for f in data['frames']]

# 绘制误差曲线
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(frames, root_pos_errors)
plt.xlabel('Frame')
plt.ylabel('Root Position Error (m)')
plt.title('Root Position Error Over Time')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(frames, joint_pos_errors)
plt.xlabel('Frame')
plt.ylabel('Joint Position Error (rad)')
plt.title('Joint Position Error Over Time')
plt.grid(True)

plt.tight_layout()
plt.savefig('./replay_debug_logs/error_analysis.png')
plt.show()

# 找出误差突然增大的帧
threshold = 0.05  # 5cm
problem_frames = [f for f, e in zip(frames, root_pos_errors) if e > threshold]
print(f"Frames with root position error > {threshold}m: {problem_frames}")
```

## Debug流程建议

### 1. 运行replay并生成日志

```bash
bash run_replay.sh
```

### 2. 查看摘要报告

```bash
cat ./replay_debug_logs/<recording_name>_summary.txt
```

找出误差最大的帧号，例如Frame 90。

### 3. 查看详细日志

```bash
# 查看Frame 90附近的详细信息
grep -A 30 "Frame 90" ./replay_debug_logs/<recording_name>.txt
```

### 4. 分析误差模式

- **误差逐渐累积**：可能是物理参数不匹配（摩擦、质量等）
- **某一帧突然增大**：可能是接触事件（碰撞、滑动）
- **周期性波动**：可能是控制器参数问题

### 5. 针对性调整

根据摘要中的建议：

- **Root位置误差大**：检查初始状态、物理参数、接触模型
- **Root速度误差大**：检查摩擦系数、地面接触
- **关节位置误差大**：检查PD控制器参数、关节限位

### 6. 可视化分析（可选）

使用上面的Python脚本绘制误差曲线，直观看出误差变化趋势。

## 常见问题

### Q: 日志文件太大怎么办？

A: 可以只记录部分帧：

```python
# 在 _log_state_comparison 方法中添加条件
if frame_idx % 10 == 0:  # 每10帧记录一次
    self.debug_logger.log_frame(frame_idx, recorded, simulated)
```

### Q: 如何只关注特定的状态？

A: 修改 `replay_debug_logger.py` 中的 `_write_frame_log` 方法，注释掉不需要的部分。

### Q: 误差单位是什么？

A:
- 位置：米 (m)
- 速度：米/秒 (m/s)
- 角度：弧度 (rad)
- 角速度：弧度/秒 (rad/s)

## 性能影响

- 文本日志写入：每帧约 0.1-0.2ms
- JSON数据存储：内存中累积，最后一次性写入
- 总体影响：replay速度降低约 5-10%

如果需要最大性能，可以禁用debug logging。
