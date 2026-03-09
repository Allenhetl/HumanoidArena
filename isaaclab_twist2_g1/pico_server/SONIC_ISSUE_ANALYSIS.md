# SONIC 推理问题分析报告

## 问题描述

`sonic_targets` 输出始终为固定的默认站立姿态：
```
[-0.312 -0.312  0.     0.     0.     0.     0.     0.     0.     0.669
  0.669  0.2    0.2   -0.363 -0.363  0.2   -0.2    0.     0.     0.
  0.     0.6    0.6    0.     0.     0.     0.     0.     0.   ]
```

这正好是 `SONIC_DEFAULT_POS`，说明推理没有正常工作。

## 诊断结果

### 1. ZMQ数据接收 ✓ 正常

```
✓ SMPL数据接收正常
  - smpl_joints: shape=(5, 24, 3)
  - 绝对值和: 21.5584 (有效数据)
  - FPS: 36.27

✓ body_quat_w: shape=(5, 4)  ← 注意：只有单个四元数
✓ joint_pos: shape=(5, 29)
```

### 2. 数据格式问题 ⚠

**发现的问题**：
- Pico服务器发送的 `body_quat_w` 形状是 `(N, 4)`
- 这只是**全局方向四元数**（`global_orient_quat`）
- **不是**24个身体部位的四元数 `(N, 24, 4)`

**代码位置**：
```python
# pico_server_pose_only.py:1251-1253
body_quat_np = (
    latest_data["global_orient_quat"].detach().cpu().numpy()[0].astype(np.float32)
)  # 只有 (4,) 而不是 (24, 4)
```

### 3. IsaacLab端的处理 ⚠

**action_provider_sonic.py 的问题**：

```python
# Line 301-302: 初始化
self._body_quat_buf = np.tile(
    np.array([1., 0., 0., 0.], dtype=np.float32), (10, 1))  # (10, 4)

# Line 365-369: 接收
if "body_quat_w" in data:
    bq = data["body_quat_w"].astype(np.float32)  # (N, 4)
    self._body_quat_buf = np.roll(self._body_quat_buf, -1, axis=0)
    self._body_quat_buf[-1] = bq[-1]  # (4,)

# Line 407: Encoder输入
anchor_orient = self._body_quat_buf[np.newaxis]  # (1, 10, 4)
```

**问题**：
- Encoder期望的输入可能是 `(1, 10, 24, 4)` 或其他形状
- 当前只提供了 `(1, 10, 4)`

## 对比C++实现

### C++端的数据流

从 `g1_deploy_onnx_ref.cpp` 和 `deploy.sh` 可以看到：

1. **接收ZMQ数据**：
   ```cpp
   // StreamedMotionMerger接收
   body_quat: [frame][body][w,x,y,z]  // (N, 24, 4)
   smpl_joints: [frame][joint][x,y,z] // (N, 24, 3)
   ```

2. **Encoder输入**：
   ```cpp
   // 需要确认encoder的输入格式
   // 可能需要24个身体部位的四元数
   ```

3. **推理流程**：
   ```
   ZMQ数据 → Encoder → latent → Planner → Decoder → 29维关节
   ```

### 关键差异

| 项目 | C++端 | IsaacLab端 | 状态 |
|------|-------|------------|------|
| body_quat形状 | (N, 24, 4)? | (N, 4) | ✗ 不匹配 |
| Planner | ✓ 使用 | ✗ 未使用 | ⚠ 可能影响 |
| 数据来源 | ZMQ | ZMQ | ✓ 相同 |

## 可能的原因

### 原因1: body_quat维度错误 (最可能)

**问题**：
- Encoder期望24个身体部位的四元数
- 当前只提供了1个全局方向四元数

**证据**：
- 诊断显示 `body_quat_w: shape=(5, 4)`
- C++端的 `StreamedMotionMerger` 支持 `(N, 24, 4)`

**解决方案**：
需要计算24个身体部位的四元数。但原始代码也是这样的，说明：
1. 要么原始代码也有问题
2. 要么Encoder实际上只需要全局方向四元数
3. 要么需要从其他地方获取24个四元数

### 原因2: 缺少Planner

**问题**：
- C++端使用Planner进行运动规划
- IsaacLab端直接使用Encoder-Decoder

**影响**：
- 可能导致输出不稳定或不正确

### 原因3: 模型输入格式不匹配

**需要检查**：
1. Encoder模型期望的输入形状
2. 当前提供的输入形状是否匹配

## 调试步骤

### 步骤1: 检查Encoder输入形状

```python
# 在 action_provider_sonic.py 中添加
print(f"Encoder inputs:")
for i, inp in enumerate(self._encoder.get_inputs()):
    print(f"  Input {i}: {inp.name}, shape: {inp.shape}, type: {inp.type}")
```

### 步骤2: 检查SMPL数据有效性

```python
# 在 _run_gear_sonic() 开头添加
print(f"_smpl_data_valid: {self._smpl_data_valid}")
print(f"_smpl_joints_buf sum: {np.abs(self._smpl_joints_buf[-1]).sum()}")
```

### 步骤3: 检查推理是否执行

```python
# 在 _run_gear_sonic() 中添加
print("Before encoder inference")
latent = self._encoder.run(None, enc_inputs)[0]
print(f"After encoder inference, latent shape: {latent.shape}")
```

### 步骤4: 使用调试补丁

运行生成的调试补丁：
```bash
python create_debug_patch.py
# 然后将生成的代码应用到 action_provider_sonic.py
```

## 建议的修复方案

### 方案1: 修复body_quat维度 (推荐)

如果Encoder确实需要24个身体部位的四元数：

1. **修改pico_server_pose_only.py**：
   ```python
   # 计算24个身体部位的四元数
   from gear_sonic.trl.utils.torch_transform import compute_body_quaternions_from_joints

   body_quat_np = compute_body_quaternions_from_joints(
       latest_data["smpl_joints_local"]
   ).detach().cpu().numpy()[0].astype(np.float32)  # (24, 4)
   ```

2. **修改action_provider_sonic.py**：
   ```python
   # 初始化
   self._body_quat_buf = np.tile(
       np.array([1., 0., 0., 0.], dtype=np.float32),
       (10, 24, 1)  # (10, 24, 4)
   )

   # 接收
   if "body_quat_w" in data:
       bq = data["body_quat_w"].astype(np.float32)  # (N, 24, 4)
       self._body_quat_buf = np.roll(self._body_quat_buf, -1, axis=0)
       self._body_quat_buf[-1] = bq[-1]  # (24, 4)

   # Encoder输入
   anchor_orient = self._body_quat_buf[np.newaxis]  # (1, 10, 24, 4)
   ```

### 方案2: 检查Encoder实际需求

1. 打印Encoder输入形状要求
2. 根据实际需求调整数据格式

### 方案3: 添加Planner

如果Planner是必需的：
1. 加载Planner模型
2. 在Encoder和Decoder之间添加Planner推理

## 下一步行动

1. **立即执行**：
   ```bash
   # 运行调试补丁生成器
   python create_debug_patch.py

   # 应用补丁到 action_provider_sonic.py
   # 重新运行并查看详细输出
   ```

2. **检查Encoder输入要求**：
   ```python
   # 在Python中
   import onnxruntime as ort
   encoder = ort.InferenceSession("path/to/encoder.onnx")
   for inp in encoder.get_inputs():
       print(f"{inp.name}: {inp.shape}")
   ```

3. **对比原始GR00T代码**：
   - 检查 `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
   - 查找Encoder输入的准备代码
   - 确认body_quat的实际格式

## 参考

- `pico_server_pose_only.py:1251-1253` - body_quat定义
- `action_provider_sonic.py:301-302, 365-369, 407` - body_quat使用
- `diagnose_sonic.py` - 诊断工具
- `create_debug_patch.py` - 调试补丁生成器