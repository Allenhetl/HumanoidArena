# 随机种子设置检查报告

## 检查结果

### ✅ 录制脚本 (sim_main.py)
```python
# Line 253-255
seed_value = args_cli.seed if args_cli.seed is not None else 42
env_cfg.seed = seed_value
print(f"[CONFIG] Setting environment seed: {seed_value}")
```
- 默认seed: 42
- 可通过 `--seed` 参数覆盖

### ✅ Replay脚本 (sim_main_replay.py)
```python
# Line 157-159
seed_value = args_cli.seed if args_cli.seed is not None else 42
env_cfg.seed = seed_value
print(f"[CONFIG] Setting environment seed: {seed_value}")
```
- 默认seed: 42
- 可通过 `--seed` 参数覆盖

### ✅ Action Provider (action_provider_wh_twist2_replay.py)
```python
# Line 25-36
if hasattr(args_cli, 'seed') and args_cli.seed is not None:
    seed = args_cli.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Enable deterministic mode for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
- 设置了torch, CUDA, numpy, random的seed
- 启用了PyTorch确定性模式

### ✅ 环境配置
- 任务: Isaac-Move-Football-G129-Dex3-Wholebody
- 环境类型: ManagerBasedRLEnv
- 配置类: MoveFootballG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg)
- Seed属性: `ManagerBasedEnvCfg.seed: int | None = None`

## 环境seed的应用流程

```python
# 1. 脚本设置
env_cfg.seed = 42

# 2. 环境初始化时应用
# isaaclab/envs/manager_based_env.py
if self.cfg.seed is not None:
    self.cfg.seed = self.seed(self.cfg.seed)

# 3. seed()方法实现
def seed(seed: int = -1) -> int:
    # Set seed for replicator
    rep.set_global_seed(seed)
    # Set seed for torch and other libraries
    return torch_utils.set_seed(seed)  # isaacsim.core.utils.torch.set_seed()
```

## 潜在的非确定性来源

### ⚠️ 1. PhysX物理引擎
- **问题**: PhysX可能有内部随机性
- **影响**: 接触求解、碰撞检测可能不完全确定
- **检查方法**: 查看PhysX配置中的随机性设置

### ⚠️ 2. GPU并行计算
- **问题**: CUDA操作的执行顺序可能不确定
- **影响**: 浮点运算顺序导致微小差异
- **已缓解**:
  - `torch.backends.cudnn.deterministic = True` ✅
  - `torch.backends.cudnn.benchmark = False` ✅

### ⚠️ 3. 多线程/异步操作
- **问题**: 物理仿真可能使用多线程
- **影响**: 线程调度导致不确定性
- **检查方法**: 查看PhysX的线程数设置

### ⚠️ 4. 浮点精度
- **问题**: 不同硬件的浮点运算精度可能不同
- **影响**: 累积误差
- **无法完全避免**

## 为什么仍有误差

即使设置了相同的seed，仍可能有误差的原因：

### 1. PD控制器参数不同
```
录制时的PD参数 ≠ Replay时的PD参数
→ 相同的目标位置产生不同的力矩
→ 关节运动轨迹不同
```

### 2. Decimation设置不同
```
录制时的decimation ≠ Replay时的decimation
→ 控制频率不同
→ 物理步数不同
→ 运动轨迹不同
```

### 3. 物理时间步长不同
```
录制时的physics_dt ≠ Replay时的physics_dt
→ 数值积分步长不同
→ 累积误差不同
```

### 4. PhysX求解器设置不同
```
可能不同的参数：
- position_iteration_count
- velocity_iteration_count
- contact_offset
- rest_offset
- bounce_threshold
```

## 建议的调试步骤

### Step 1: 验证seed是否生效
在录制和replay时添加日志：
```python
print(f"[DEBUG] Environment seed: {env.cfg.seed}")
print(f"[DEBUG] Torch seed: {torch.initial_seed()}")
print(f"[DEBUG] Numpy random state: {np.random.get_state()[1][0]}")
```

### Step 2: 对比物理参数
```bash
# 录制时
grep -r "decimation\|physics_dt\|position_iteration" config/

# Replay时
# 打印相同的参数
```

### Step 3: 对比PD控制器参数
```python
# 在初始化后打印
robot = env.scene["robot"]
print(f"Actuator stiffness: {robot.actuators['twist2'].stiffness}")
print(f"Actuator damping: {robot.actuators['twist2'].damping}")
print(f"Effort limit: {robot.actuators['twist2'].effort_limit}")
```

### Step 4: 测试简化场景
创建一个最小测试：
1. 设置相同的初始状态
2. 发送相同的动作
3. 运行1个decimation周期
4. 对比结果

如果简化场景也有误差，说明是物理引擎配置问题。

## 结论

✅ **Seed设置正确**:
- 录制和replay都设置了 `env_cfg.seed = 42`
- Action Provider设置了确定性模式
- 环境会应用seed到物理引擎

❌ **但仍有误差的原因**:
- 很可能是PD控制器参数、decimation或物理参数不匹配
- 不是seed的问题

🔍 **下一步**:
1. 对比录制和replay的物理参数（decimation, physics_dt）
2. 对比PD控制器参数（stiffness, damping, effort_limit）
3. 检查PhysX求解器设置
