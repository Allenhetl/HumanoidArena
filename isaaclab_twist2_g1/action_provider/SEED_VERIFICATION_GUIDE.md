# Seed和参数验证指南

## 问题确认

你提到：
- ✅ Decimation = 10
- ✅ sim.dt = 0.002
- ✅ Stiffness (Kp), Damping (Kd), Effort limit 应该是一样的

但仍然有大误差（关节18只移动了15%的距离）。

## 需要验证的内容

### 1. 环境seed是否真的被应用

**验证方法：** 运行replay时查看控制台输出

应该看到：
```
[CONFIG] Setting environment seed: 42
[INFO]: Base environment:
    Environment seed      : 42
```

如果看到 `Environment seed: None`，说明seed没有被应用。

### 2. Action Provider是否收到seed

**验证方法：** 查看控制台输出

应该看到：
```
[ReplayActionProvider] Setting random seed: 42
[ReplayActionProvider] ===== Configuration Verification =====
[ReplayActionProvider]   Seed (env): 42
[ReplayActionProvider]   Seed (action provider): 42
[ReplayActionProvider]   PyTorch deterministic: True
[ReplayActionProvider]   PyTorch benchmark: False
```

如果看到 `No seed specified, using non-deterministic mode`，说明seed没有传递给action provider。

### 3. 物理参数验证

**验证方法：** 查看控制台输出

应该看到：
```
[ReplayActionProvider]   Decimation: 10
[ReplayActionProvider]   Physics dt: 0.002
[ReplayActionProvider]   Control dt: 0.02
[ReplayActionProvider]   PD Stiffness (first 5): [...]
[ReplayActionProvider]   PD Damping (first 5): [...]
[ReplayActionProvider]   Effort limit (first 5): [...]
```

**对比录制时的参数：** 需要在录制脚本中也添加相同的日志。

## 如何添加验证日志

### 在录制脚本 (sim_main.py) 中添加

在创建action provider后添加：
```python
# 在 action_provider 创建后
print(f"[RECORDING] ===== Configuration =====")
print(f"[RECORDING]   Decimation: {env.cfg.decimation}")
print(f"[RECORDING]   Physics dt: {env.physics_dt}")
print(f"[RECORDING]   Seed: {env.cfg.seed}")

robot = env.scene["robot"]
if hasattr(robot, 'actuators') and 'twist2' in robot.actuators:
    actuator = robot.actuators['twist2']
    print(f"[RECORDING]   PD Stiffness (first 5): {actuator.stiffness[:5]}")
    print(f"[RECORDING]   PD Damping (first 5): {actuator.damping[:5]}")
    print(f"[RECORDING]   Effort limit (first 5): {actuator.effort_limit[:5]}")
print(f"[RECORDING] ==============================")
```

## 可能的问题和解决方案

### 问题1: Seed没有传递给Action Provider

**症状：** 看到 `No seed specified, using non-deterministic mode`

**原因：** `args_cli.seed` 是 `None`

**解决方案：**
```bash
# 运行replay时显式指定seed
python sim_main_replay.py --task Isaac-Move-Football-G129-Dex3-Wholebody --replay_file xxx.npz --seed 42
```

### 问题2: 环境seed没有被应用

**症状：** 看到 `Environment seed: None`

**原因：** `env_cfg.seed` 没有被设置

**解决方案：** 检查 `sim_main_replay.py` 中的代码：
```python
seed_value = args_cli.seed if args_cli.seed is not None else 42
env_cfg.seed = seed_value
```

### 问题3: PD参数不同

**症状：** 录制和replay的PD参数不一致

**原因：** 使用了不同的机器人配置文件

**解决方案：** 确保使用相同的配置文件

### 问题4: 即使参数相同仍有误差

**可能原因：**

1. **PhysX内部随机性**
   - PhysX可能有无法通过seed控制的随机性
   - 特别是接触求解器

2. **浮点精度累积误差**
   - 即使初始状态相同，浮点运算的微小差异会累积
   - 特别是在多步仿真后

3. **GPU并行计算的不确定性**
   - 即使设置了 `cudnn.deterministic = True`
   - 某些CUDA操作仍可能不完全确定

4. **初始状态设置时机**
   - 如果初始状态在第一个physics step之后才设置
   - 可能已经产生了初始误差

## 测试步骤

### Step 1: 运行replay并收集日志
```bash
python sim_main_replay.py \
    --task Isaac-Move-Football-G129-Dex3-Wholebody \
    --replay_file recording_data/xxx.npz \
    --seed 42 \
    > replay_log.txt 2>&1
```

### Step 2: 检查日志中的关键信息
```bash
grep -E "seed|Seed|Decimation|Physics dt|Stiffness|Damping" replay_log.txt
```

### Step 3: 对比录制时的参数
如果有录制时的日志，对比：
- Decimation
- Physics dt
- PD Stiffness
- PD Damping
- Effort limit

### Step 4: 如果参数都相同但仍有误差

这说明问题不在参数配置，而在于：
1. **物理引擎的内在不确定性**
2. **初始状态设置的时机问题**
3. **数值精度累积误差**

**可能的解决方案：**
- 接受一定的误差范围（例如 <0.1 rad）
- 使用更精确的初始状态（包括加速度等）
- 减小physics dt以提高精度
- 增加solver迭代次数

## 预期结果

如果一切设置正确：
- Frame 1的误差应该 <0.05 rad（关节位置）
- Frame 1的误差应该 <2.0 rad/s（关节速度）
- 误差不应该快速累积

如果误差仍然很大（如关节18的0.228 rad），说明：
- 初始状态没有正确设置，或
- 物理参数确实不同，或
- 存在其他未知的差异

## 下一步

1. **运行replay并查看控制台输出**
2. **检查所有验证信息是否正确**
3. **如果发现参数不一致，调整配置**
4. **如果参数一致但仍有误差，需要深入调查物理引擎**
