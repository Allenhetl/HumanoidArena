# Replay Debug System

用于对比录制数据和replay仿真状态，定位replay失败原因的调试工具。

## 快速开始

```bash
# 1. 运行replay（自动生成debug日志）
bash run_replay.sh

# 2. 等待replay完成，看到保存提示
# [ReplayActionProvider] Replay completed, saving debug logs...
# [ReplayDebugLogger] 日志已关闭，共记录 150 帧
# ✅ Debug logs saved successfully

# 3. 查看摘要（找出问题帧）
cat ./replay_debug_logs/*_summary.txt

# 4. 查看详细日志（定位具体问题）
grep -A 30 "Frame 90" ./replay_debug_logs/*.txt
```

**重要特性**：
- ✅ 日志在**replay完成时自动保存**（不需要等IsaacLab关闭）
- ✅ 即使直接关闭窗口，已记录的数据也会保存
- ✅ 支持双重保护：replay完成时保存 + cleanup时保存

## 生成的文件

运行replay后，会在 `./replay_debug_logs/` 目录下生成3个文件：

| 文件 | 用途 | 如何使用 |
|------|------|----------|
| `*.txt` | 逐帧详细对比 | 用 `grep` 查找特定帧 |
| `*_summary.txt` | 摘要报告 | **推荐先看这个** |
| `*.json` | 结构化数据 | 用于分析脚本 |

## 使用流程

### 方式1: 纯文本分析（推荐，无需额外依赖）

```bash
# 步骤1: 查看摘要，找出最大误差的帧
cat ./replay_debug_logs/your_recording_summary.txt

# 输出示例：
# 最大误差 (L2范数):
#   root_pos      : 0.123456  (Frame 92)  ← 问题帧
#   root_lin_vel  : 0.234567  (Frame 90)
#   joint_pos     : 0.156789  (Frame 93)

# 步骤2: 查看问题帧的详细信息
grep -A 50 "Frame 92" ./replay_debug_logs/your_recording.txt

# 步骤3: 根据误差类型调试
# - root_pos误差大 → 检查物理参数、接触模型
# - root_lin_vel误差大 → 检查摩擦系数、地面接触
# - joint_pos误差大 → 检查PD控制器参数
```

### 方式2: 可视化分析（需要matplotlib）

```bash
# 使用JSON文件生成误差曲线图
python action_provider/analyze_replay_errors.py \
    ./replay_debug_logs/your_recording.json

# 生成：
# - error_analysis.png (误差曲线图)
# - error_analysis_report.txt (分析报告)
```

## 文档

- **[REPLAY_DEBUG_QUICKSTART.md](./REPLAY_DEBUG_QUICKSTART.md)** - 快速开始指南
- **[REPLAY_DEBUG_USAGE.md](./REPLAY_DEBUG_USAGE.md)** - 详细使用说明

## 配置选项

### 禁用debug logging

```bash
python sim_main_replay.py \
    --replay_file ./recording_data/your_recording.npz \
    --enable_replay_debug_log False
```

### 自定义日志目录

```bash
python sim_main_replay.py \
    --replay_file ./recording_data/your_recording.npz \
    --replay_debug_log_dir ./my_debug_logs
```

## 常见问题

### Q: 日志文件太大怎么办？

A: 可以只记录部分帧，修改 `action_provider_wh_twist2_replay.py`:

```python
# 在 _log_state_comparison 方法开头添加
if frame_idx % 10 == 0:  # 每10帧记录一次
    self.debug_logger.log_frame(frame_idx, recorded, simulated)
```

### Q: 分析脚本需要什么依赖？

A: 需要 `matplotlib` 和 `numpy`：

```bash
pip install matplotlib numpy
```

如果没有matplotlib，可以只看文本日志，功能完全够用。

### Q: 误差单位是什么？

A:
- 位置：米 (m)
- 速度：米/秒 (m/s)
- 角度：弧度 (rad)
- 角速度：弧度/秒 (rad/s)

### Q: 如何判断误差是否正常？

A: 参考阈值：
- Root位置误差 < 0.05m (5cm) - 正常
- Root速度误差 < 0.2m/s - 正常
- 关节位置误差 < 0.1rad (5.7度) - 正常

如果超过这些阈值，说明replay有问题。

## 性能影响

- 日志记录开销：每帧约 0.1-0.2ms
- Replay速度降低：约 5-10%
- 内存占用：每帧约 2KB（JSON数据）

如果需要最大性能，可以禁用debug logging。

## 示例输出

### 摘要报告示例

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
```

### 详细日志示例

```
====================================================================================================
Frame 90
====================================================================================================

Root Position:
  Recorded:  [1.234567, -0.123456, 0.987654] m
  Simulated: [1.234500, -0.123400, 0.987600] m
  Error (L2): 0.000123 m
  Error (Max): 0.000067 m

Root Linear Velocity:
  Recorded:  [0.523456, -0.123456, 0.012345] m/s
  Simulated: [0.489012, -0.098765, 0.015678] m/s  ← 仿真速度偏小
  Error (L2): 0.234567 m/s
  Error (Max): 0.034444 m/s

Joint Positions (29 DOFs):
  Error (L2): 0.012345 rad
  Error (Max): 0.005678 rad (joint 15)
  Error (Mean): 0.000234 rad
```

## 工作原理

1. **数据收集**: 每帧记录录制数据和仿真状态
2. **误差计算**: 计算L2范数、最大值、平均值
3. **实时写入**: 详细日志实时写入txt文件
4. **内存缓存**: JSON数据在内存中累积
5. **统一输出**: 程序结束时生成JSON和摘要

## 相关文件

- `replay_debug_logger.py` - 日志记录器实现
- `analyze_replay_errors.py` - 误差分析脚本
- `action_provider_wh_twist2_replay.py` - 集成了logger的replay代码
