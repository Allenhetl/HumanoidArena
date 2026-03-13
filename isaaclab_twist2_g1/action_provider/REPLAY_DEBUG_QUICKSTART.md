# Replay Debug 快速开始

## 1. 运行replay（自动生成debug日志）

```bash
bash run_replay.sh
```

或者：

```bash
python sim_main_replay.py \
    --replay_file ./recording_data/your_recording.npz \
    --replay_mode direct
```

**自动生成的文件**：
- `your_recording.txt` - 详细的逐帧对比日志（人类可读）
- `your_recording.json` - 结构化数据（用于分析脚本）
- `your_recording_summary.txt` - 摘要报告（推荐先看这个）

**⚠️ 重要**：
- 日志会在**replay完成时自动保存**（不需要等IsaacLab关闭）
- 看到 `✅ Debug logs saved successfully` 表示保存成功
- 即使直接关闭IsaacLab窗口，已记录的数据也会保存

## 2. 查看摘要（找出最大误差的帧）

```bash
cat ./replay_debug_logs/your_recording_summary.txt
```

输出示例：
```
最大误差 (L2范数):
  root_pos            : 0.123456  (Frame 92)  ← 在Frame 92出现最大位置误差
  root_lin_vel        : 0.234567  (Frame 90)  ← 在Frame 90出现最大速度误差
  joint_pos           : 0.156789  (Frame 93)  ← 在Frame 93出现最大关节误差
```

## 3. 查看详细日志（定位具体问题）

```bash
# 查看Frame 90附近的详细信息
grep -A 30 "Frame 90" ./replay_debug_logs/your_recording.txt
```

## 4. 可视化分析（可选，需要matplotlib）

**使用JSON文件进行分析**：
```bash
# 注意：使用 .json 文件，不是 .txt 文件
python action_provider/analyze_replay_errors.py \
    ./replay_debug_logs/your_recording.json
```

会生成：
- `error_analysis.png` - 误差曲线图（6个子图）
- `error_analysis_report.txt` - 详细分析报告

**如果没有matplotlib**：
```bash
# 只看文本日志也足够了
cat ./replay_debug_logs/your_recording_summary.txt
grep -A 30 "Frame 90" ./replay_debug_logs/your_recording.txt
```

## 5. 根据误差类型调试

### 如果 root_pos 误差大：
- 检查初始状态设置是否正确
- 检查物理参数（质量、惯性）
- 检查接触模型（摩擦系数）

### 如果 root_lin_vel 误差大：
- 检查地面摩擦系数
- 检查接触力模型
- 检查质量分布

### 如果 joint_pos 误差大：
- 检查PD控制器参数（kp, kd）
- 检查关节限位
- 检查动作目标是否正确

## 6. 对比特定帧

```bash
# 查看Frame 90的详细对比
grep -A 50 "^Frame 90$" ./replay_debug_logs/your_recording.txt | less
```

## 常见问题排查

### Q: 误差从某一帧突然增大
**可能原因**: 接触事件（碰撞、滑动、失去平衡）
**排查方法**:
1. 查看该帧的详细日志
2. 检查root速度是否突变
3. 检查是否有关节达到限位

### Q: 误差逐渐累积
**可能原因**: 物理参数不匹配
**排查方法**:
1. 对比录制和replay的物理参数
2. 检查摩擦系数、质量、惯性
3. 检查PD控制器参数

### Q: 误差周期性波动
**可能原因**: 控制器参数问题
**排查方法**:
1. 使用分析脚本检测周期
2. 调整PD控制器的kp/kd
3. 检查decimation设置

## 禁用debug logging（如果需要）

```bash
python sim_main_replay.py \
    --replay_file ./recording_data/your_recording.npz \
    --replay_mode direct \
    --enable_replay_debug_log False
```

## 文件说明

**自动生成的3个文件**：
1. `your_recording.txt` - 逐帧详细对比（可能很大，用grep查找）
2. `your_recording_summary.txt` - 摘要报告（**推荐先看这个**）
3. `your_recording.json` - 结构化数据（**用于分析脚本**）

**运行分析脚本后生成**：
4. `error_analysis.png` - 误差曲线图（需要matplotlib）
5. `error_analysis_report.txt` - 分析报告

**使用建议**：
- 快速定位问题：看 `_summary.txt`
- 查看详细对比：用 `grep` 搜索 `.txt` 文件
- 可视化分析：用 `.json` 文件运行分析脚本
