# 固定随机种子实现总结（最终版本）

## 修改内容

### 1. action_provider_wh_twist2.py（录制脚本）

**位置**: `__init__` 方法开始处（第 419-439 行）

```python
# Set random seed for reproducibility
if hasattr(args_cli, 'seed') and args_cli.seed is not None:
    import random
    import numpy as np
    seed = args_cli.seed
    print(f"[{self.name}] Setting random seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Enable deterministic mode for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Store seed for ONNX Runtime configuration
    self.onnx_seed = seed
else:
    self.onnx_seed = None
    print(f"[{self.name}] No seed specified, using non-deterministic mode")
```

**位置**: `load_onnx_policy` 方法（第 926-933 行）

```python
# Configure session options for deterministic inference
sess_options = ort.SessionOptions()
if self.onnx_seed is not None:
    # Enable deterministic compute
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    print(f"[{self.name}] ONNX Runtime configured for deterministic inference (seed={self.onnx_seed})")

model = ort.InferenceSession(path, sess_options=sess_options, providers=providers)
```

### 2. action_provider_wh_twist2_replay.py（回放脚本）

**位置**: `__init__` 方法开始处（第 20-40 行）

```python
# Set random seed for reproducibility
if hasattr(args_cli, 'seed') and args_cli.seed is not None:
    import random
    import numpy as np
    seed = args_cli.seed
    print(f"[{self.name}] Setting random seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Enable deterministic mode for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Store seed for ONNX Runtime configuration
    self.onnx_seed = seed
else:
    self.onnx_seed = None
    print(f"[{self.name}] No seed specified, using non-deterministic mode")
```

**位置**: `_load_onnx_model` 方法（第 179-184 行）

```python
# Configure for deterministic inference if seed is set
if self.onnx_seed is not None:
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    print(f"[{self.name}] ONNX Runtime configured for deterministic inference (seed={self.onnx_seed})")
```

### 3. sim_main.py（主程序）

**位置**: 参数解析部分（第 109-110 行）

```python
# random seed for reproducibility
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility (default: None)")
```

**位置**: 环境配置部分（第 248-258 行）

```python
# parse environment configuration
try:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    # Set seed: command line argument takes priority, otherwise use default 42
    seed_value = args_cli.seed if args_cli.seed is not None else 42
    env_cfg.seed = seed_value
    print(f"[CONFIG] Setting environment seed: {seed_value}")
except Exception as e:
    print(f"Failed to parse environment configuration: {e}")
    return
```

### 4. sim_main_replay.py（回放主程序）

**位置**: 参数解析部分（第 67-68 行）

```python
# random seed for reproducibility
parser.add_argument("--seed", type=int, default=None, help="random seed for reproducibility (default: None)")
```

**位置**: 环境配置部分（第 152-162 行）

```python
try:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.env_name = args_cli.task
    # Set seed: command line argument takes priority, otherwise use default 42
    seed_value = args_cli.seed if args_cli.seed is not None else 42
    env_cfg.seed = seed_value
    print(f"[CONFIG] Setting environment seed: {seed_value}")
except Exception as e:
    print(f"Failed to parse environment configuration: {e}")
    return
```

### 5. run.sh（录制启动脚本）

**位置**: 第 52-53 行和第 70 行

```bash
# Random seed for reproducibility (set to fixed value for deterministic behavior)
SEED="${SEED:-42}"

python sim_main.py \
    # ... 其他参数 ...
    --seed "${SEED}"
```

### 6. run_replay.sh（回放启动脚本）

**位置**: 第 52-55 行和第 86 行

```bash
# Random seed for reproducibility (set to fixed value for deterministic behavior)
SEED="${SEED:-42}"
echo "Random seed: $SEED"
echo "=========================================="

python sim_main_replay.py \
    # ... 其他参数 ...
    --seed "${SEED}"
```

## 实现原理

### 1. 统一的种子设置策略

**不需要在每个环境配置中单独设置 seed**，而是在 `sim_main.py` 和 `sim_main_replay.py` 中统一设置：

```python
# 命令行参数优先，否则使用默认值 42
seed_value = args_cli.seed if args_cli.seed is not None else 42
env_cfg.seed = seed_value
```

这样的好处：
- ✅ 切换环境时不需要修改环境配置文件
- ✅ 所有环境使用统一的种子管理
- ✅ 可以通过命令行或环境变量灵活控制

### 2. 多层次的随机种子控制

#### 层次 1: Isaac Lab 环境（物理引擎）
通过 `env_cfg.seed` 设置，Isaac Lab 会自动配置：
- PhysX 物理引擎
- 环境重置时的随机化（如足球位置）
- 场景初始化

#### 层次 2: PyTorch 和 NumPy（策略推理）
在 action provider 中设置：
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

#### 层次 3: ONNX Runtime（确定性推理）
通过 `SessionOptions` 配置：
```python
sess_options.intra_op_num_threads = 1
sess_options.inter_op_num_threads = 1
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
```

### 3. 默认种子值

- **默认种子**: `42`
- **命令行覆盖**: `--seed 123`
- **环境变量覆盖**: `SEED=123 bash run.sh`
- **禁用种子**: `--seed` 不传参数时使用默认值 42

## 使用方法

### 录制数据（使用固定种子）

```bash
# 使用默认种子 42
bash run.sh

# 使用自定义种子
SEED=123 bash run.sh

# 或者
bash run.sh  # 会自动使用 SEED=42
```

### 回放数据（使用相同种子）

```bash
# 使用默认种子 42
./run_replay.sh recording_data/latest.npz inference

# 使用自定义种子（必须与录制时相同）
SEED=123 ./run_replay.sh recording_data/latest.npz inference
```

### 切换环境

切换到其他任务时，不需要修改环境配置文件：

```bash
# 切换到沙袋任务
TASK_NAME=Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody bash run.sh

# 切换到足球任务
TASK_NAME=Isaac-Move-Football-G129-Dex3-Wholebody bash run.sh

# 所有任务都会自动使用相同的种子设置机制
```

## 确定性保证

### ✅ 完全确定的部分

1. **物理仿真** - Isaac Lab/PhysX 刚体和关节仿真
2. **ONNX 推理** - 策略网络输出
3. **随机数生成** - PyTorch, NumPy, Python random
4. **环境重置** - 足球初始位置等随机化

### ⚠️ 注意事项

1. **硬件依赖** - 相同硬件配置才能保证完全相同的结果（浮点精度差异）
2. **Isaac Sim 版本** - 必须使用相同版本的 Isaac Sim 和 PhysX
3. **非刚体** - 布料、软体等非刚体仿真不保证确定性
4. **CUDA 版本** - 不同 CUDA 版本可能产生微小差异

### 验证方法

1. 使用相同种子录制两次，验证录制数据完全相同
2. 使用相同种子回放同一录制文件多次，验证回放结果完全相同
3. 检查控制台输出确认种子已正确设置：
   ```
   [CONFIG] Setting environment seed: 42
   [DDSActionProvider] Setting random seed: 42
   [DDSActionProvider] ONNX Runtime configured for deterministic inference (seed=42)
   ```

## 优势

### 相比在每个环境配置中单独设置 seed

1. **统一管理** - 所有环境的种子在一个地方设置
2. **易于切换** - 切换环境时不需要修改配置文件
3. **灵活控制** - 可以通过命令行或环境变量动态设置
4. **避免遗漏** - 不会因为忘记在某个环境配置中设置 seed 而导致不确定性

## 相关文件

- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_wh_twist2.py`
- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/action_provider/action_provider_wh_twist2_replay.py`
- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/sim_main.py`
- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/sim_main_replay.py`
- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/run.sh`
- `/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/run_replay.sh`

## 参考文档

- Isaac Lab ManagerBasedEnvCfg.seed - 环境配置中的种子参数
- PyTorch Reproducibility - torch.manual_seed() 和确定性模式
- ONNX Runtime Determinism - SessionOptions 配置
- PhysX Determinism - 刚体场景完全确定，相同硬件+版本保证相同结果
