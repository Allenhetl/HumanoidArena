## SONIC Latent不变问题 - 完整修复总结

### 问题现象
在VR中规律甩手，IsaacLab仿真中机器人有动作但不跟随VR运动，没有规律性。

### 发现的两个Bug

#### Bug 1: Encoder输入字段全0
**现象**：Latent输出不变（0.000000），即使SMPL数据正常变化

**根因**：在`action_provider_sonic.py`的`_run_gear_sonic()`中，encoder输入的机器人状态字段全是0：
```python
# 计算了但没用
motion_joint_pos_step5_ref = gather_temporal_window(...)

# 实际使用的是全0！
motion_joint_pos_step5_full = np.zeros((_STEP5_FRAMES * 29,), dtype=np.float32)
```

**修复**：用实际机器人状态填充encoder输入
```python
robot_joint_pos_step5 = gather_temporal_window(
    self._robot_joint_pos_hist, _STEP5_FRAMES, _STEP5_STRIDE
)
motion_joint_pos_step5_full = robot_joint_pos_step5.reshape(-1).astype(np.float32)
```

#### Bug 2: 历史缓冲区太短
**现象**：修复Bug 1后，运行时报错
```
ValueError: History too short for num_frames=10, stride=5: len=10, required=46
```

**根因**：`_robot_joint_pos_hist`只有10帧，但step5采样需要46帧

**修复**：扩大历史缓冲区到46帧
```python
# 修复前：10帧
self._robot_joint_pos_hist = np.tile(..., (_STEP1_FRAMES, 1))  # (10, 29)

# 修复后：46帧
self._robot_joint_pos_hist = np.tile(..., (_STEP5_HISTORY_LEN, 1))  # (46, 29)
```

同时在decoder输入构建时，从46帧中提取最近10帧：
```python
robot_joint_pos_step1 = gather_temporal_window(
    self._robot_joint_pos_hist, _STEP1_FRAMES, 1
)
```

### 修改的文件

`action_provider/action_provider_sonic.py`：
1. 第505-510行：扩大历史缓冲区从10帧到46帧
2. 第837-868行：用机器人实际状态填充encoder输入
3. 第971-987行：从46帧历史中提取10帧用于decoder输入

### 验证结果

✅ **Latent变化测试** (`tools/verify_fix.py`)
- 帧1→2 Latent变化：0.062500
- 帧2→3 Latent变化：0.062500

✅ **历史缓冲区测试** (`tools/verify_history_fix.py`)
- step5采样（10帧，stride=5）：成功
- step1采样（10帧，stride=1）：成功
- 采样正确性验证：正确

### 测试步骤

1. 启动pico_server（gear_sonic侧）
2. 启动IsaacLab仿真：`./run.sh`
3. 在VR中规律甩手（左右甩，约1Hz）
4. 观察仿真中机器人是否跟随VR运动

**预期结果**：机器人手臂跟随VR规律摆动，有明显的左右摆动规律性

### 关键洞察

1. **SONIC encoder是条件编码器**
   - 需要SMPL数据（目标姿态）+ 机器人当前状态
   - 只提供SMPL而其他字段全0 → encoder产生固定latent
   - 类似条件VAE/GAN，条件和目标都必须有合理值

2. **历史缓冲区大小要求**
   - step5采样：`(10-1)*5+1=46`帧
   - step1采样：`(10-1)*1+1=10`帧
   - 必须使用足够大的缓冲区支持最大采样需求

### 调试工具

- `tools/verify_fix.py` - 验证latent变化
- `tools/verify_history_fix.py` - 验证历史缓冲区
- `tools/test_fix.sh` - 测试指南
- `tools/diagnose_sonic_dataflow.py` - 完整数据流诊断
