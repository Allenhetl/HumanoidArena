# Replay不一致问题分析报告

## 问题描述

在使用replay脚本回放录制数据时，发现：
1. **Direct模式**：直接使用录制的qpos，误差相对较小但仍然存在
2. **Inference模式**：使用ONNX模型重新推理，误差很大且快速累积

## 根本原因

### 1. 时序不匹配问题

**录制时的数据采集时序**：
```
Frame N (get_action开始):
  ├─ 1. 读取当前状态 (上一帧物理仿真后的结果)
  │    ├─ root_position: [-1.915, -5.171, 0.783]
  │    ├─ root_orientation: [0.703, -0.017, 0.014, 0.711]
  │    └─ joint_pos: [29个关节位置]
  │
  ├─ 2. 构建observation (包含当前状态 + 历史)
  │    └─ obs_buf: [1432维] = [obs_full(127) + obs_hist(1270) + future(35)]
  │
  ├─ 3. ONNX推理
  │    └─ action: [29维] → target_29 = action * 0.5 + default_pos
  │
  ├─ 4. 保存录制数据 ← 保存的是步骤1读取的状态！
  │    ├─ robot_root_position: [-1.915, -5.171, 0.783]
  │    ├─ robot_twist2_inference_qpos: target_29
  │    └─ robot_obs_buf: obs_buf
  │
  └─ 5. 执行物理仿真 (decimation=4步)
       └─ root状态变化: ~0.018m/frame
```

**Replay时的执行时序**：
```
Frame N (get_action开始):
  ├─ 1. 从录制数据加载
  │    ├─ root_pos = recorded_root_position[N]  ← 这是录制时Frame N开始的状态
  │    └─ target_29 = recorded_qpos[N]
  │
  ├─ 2. 设置root状态
  │    └─ write_root_pose_to_sim(root_pos, root_quat)
  │
  ├─ 3. 设置关节目标
  │    └─ set_joint_position_target(target_29)
  │
  └─ 4. 执行物理仿真 (decimation=4步)
       └─ root状态变化: ~0.018m/frame ← 但起点已经不同！
```

**关键问题**：
- 录制时保存的root状态是**Frame N开始时的状态**（上一帧仿真后的结果）
- Replay时我们设置这个状态，然后执行物理仿真
- 但物理仿真会根据当前的joint positions和velocities继续演化
- 由于微小的数值误差，演化结果会逐渐偏离录制轨迹

### 2. Direct模式误差分析

**误差来源**：
1. **浮点数精度**：CPU/GPU浮点运算的微小差异
2. **物理引擎数值积分**：Runge-Kutta等积分方法的截断误差
3. **碰撞检测**：接触力计算的微小差异
4. **Root状态设置**：强制设置root状态会打断物理连续性

**误差累积**：
```
Frame 0:   root误差 = 0
Frame 1:   root误差 ≈ 0.001m  (设置误差 + 物理误差)
Frame 2:   root误差 ≈ 0.002m  (累积)
...
Frame 396: root误差 ≈ 0.5-1.0m (显著偏离)
```

**数据支持**：
- 平均每帧root位移：0.018m
- 396帧总位移：2.69m
- 如果每帧有0.1%的误差，396帧后累积误差 ≈ 0.27m (10%偏差)

### 3. Inference模式误差更大的原因

**误差传播链**：
```
Frame N:
  微小root误差 (0.001m)
    ↓
  observation偏差 (L2 norm ≈ 0.1)
    ↓
  ONNX输出不同action (Δaction ≈ 0.05 rad)
    ↓
Frame N+1:
  更大的root误差 (0.002m)
    ↓
  更大的observation偏差 (L2 norm ≈ 0.2)
    ↓
  更不同的action (Δaction ≈ 0.1 rad)
    ↓
Frame N+2:
  指数级增长...
```

**关键数据**：
- Observation每帧变化：L2 norm = 10.4 (非常大！)
- Joint position每帧变化：1.27 rad (也很大)
- 这意味着系统对observation非常敏感

**为什么误差会指数增长**：
1. **Observation包含历史**：10帧历史 = 1270维数据
2. **历史数据累积误差**：每帧的误差都会进入历史缓冲区
3. **模型对历史敏感**：TWIST2模型依赖历史来预测动作
4. **正反馈循环**：错误的action → 错误的状态 → 更错误的observation → 更错误的action

## 为什么录制时看起来正常？

**录制时的"自洽性"**：
```
真实物理仿真 → 真实状态 → 真实observation → 模型推理 → action
                    ↓
                物理引擎自然演化 (连续、平滑)
                    ↓
                下一帧真实状态 (自洽)
```

**Replay时的"不自洽"**：
```
强制设置状态 → 不连续 → 物理引擎尝试修正 → 产生误差
     ↓
录制的observation (基于录制时的历史)
     ↓
模型推理 (基于不匹配的observation)
     ↓
action与当前真实状态不匹配
     ↓
误差累积
```

## 解决方案

### 方案1：POSITION_ONLY模式（最稳定）
**策略**：只设置关节位置，完全不干预root状态

**优点**：
- 物理连续性最好
- 不会有jarring/抖动
- 长期稳定

**缺点**：
- Root轨迹会偏离录制轨迹
- 不适合需要精确复现的场景

**适用场景**：
- 验证关节动作是否合理
- 测试策略的鲁棒性

### 方案2：PERIODIC_CORRECTION模式（推荐）
**策略**：每N帧（如10帧）校正一次root状态

**优点**：
- 平衡精度和稳定性
- 减少jarring
- 误差不会无限累积

**缺点**：
- 仍有周期性的小跳变
- 需要调整correction_interval参数

**适用场景**：
- 大多数replay场景
- 需要较准确复现但允许小误差

**实现**：
```python
if self.current_frame % correction_interval == 0:
    # 校正root状态
    self.env.scene["robot"].write_root_pose_to_sim(recorded_root_pose)
```

### 方案3：ROOT_VELOCITY模式（最平滑）
**策略**：设置root速度而不是位置

**优点**：
- 最平滑，无jarring
- 物理上更合理
- 误差累积较慢

**缺点**：
- 需要计算速度（数值微分）
- 长期仍会有累积误差

**适用场景**：
- 需要平滑replay
- 可以容忍轨迹偏移

**实现**：
```python
root_vel = (pos[t] - pos[t-1]) / dt
self.env.scene["robot"].write_root_velocity_to_sim(root_vel)
```

### 方案4：FULL_STATE模式（当前实现）
**策略**：每帧都设置完整的root状态

**优点**：
- 最接近录制轨迹
- 短期精度最高

**缺点**：
- 可能有jarring/抖动
- 打断物理连续性
- 对物理引擎要求高

**适用场景**：
- 短片段replay
- 需要精确复现特定帧

## 推荐配置

### Direct模式
```python
# 推荐使用PERIODIC_CORRECTION
replay_mode = "direct"
drift_correction_strategy = "PERIODIC_CORRECTION"
correction_interval = 10  # 每10帧校正一次
```

### Inference模式
```python
# 由于误差累积快，建议更频繁校正
replay_mode = "inference"
drift_correction_strategy = "PERIODIC_CORRECTION"
correction_interval = 5  # 每5帧校正一次

# 或者使用POSITION_ONLY，完全依赖模型
drift_correction_strategy = "POSITION_ONLY"
```

## 进一步改进方向

1. **录制root velocity而不是position**
   - 修改录制脚本，保存root_lin_vel和root_ang_vel
   - Replay时设置速度而不是位置

2. **使用Kalman滤波融合**
   - 融合录制状态和物理仿真状态
   - 平滑过渡，减少jarring

3. **分段replay**
   - 将长序列分成多个短段
   - 每段开始时重置环境
   - 避免长期累积误差

4. **添加状态监控**
   - 实时监控replay误差
   - 当误差超过阈值时自动校正

## 总结

**核心问题**：Replay试图"强制"机器人回到录制状态，但物理引擎会根据当前状态继续演化，导致不可避免的误差累积。

**根本矛盾**：
- 精确复现 vs 物理连续性
- 短期精度 vs 长期稳定性

**最佳实践**：
- Direct模式：使用PERIODIC_CORRECTION，correction_interval=10
- Inference模式：使用PERIODIC_CORRECTION，correction_interval=5，或POSITION_ONLY
- 长序列：考虑分段replay
- 关键帧：使用FULL_STATE
