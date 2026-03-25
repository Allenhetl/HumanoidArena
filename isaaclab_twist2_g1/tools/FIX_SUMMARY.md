## SONIC Latent不变问题 - 根因分析与修复

### 问题现象
在VR中规律甩手，IsaacLab仿真中机器人有动作但不跟随VR运动，没有规律性。

### 调试过程

1. **确认SMPL数据流正常**
   - ZMQ接收到的SMPL数据正常变化（0.007-0.295）
   - SMPL数据正确放置在encoder输入的922-1642位置
   - ✅ SMPL数据流没有问题

2. **发现Latent不变**
   - 添加详细调试日志后发现：
     - SMPL输入变化：0.007-0.295
     - **Latent输出不变：0.000000**
     - Action输出变化：3.5-8.4（但因latent不变，变化无规律）
   - ❌ 问题定位：encoder输出的latent完全不变

3. **测试encoder模型**
   - 创建test_encoder_with_real_data.py测试encoder模型
   - 发现：用不同输入测试，encoder能产生不同输出
   - ✅ encoder模型本身没问题

4. **找到根本原因**
   - 创建test_encoder_with_realistic_input.py
   - 用合理的非零值填充encoder输入的所有字段
   - 结果：**Latent正常变化！**
   - ✅ 根因：encoder输入的其他字段全是0，导致encoder忽略SMPL数据

### 根本原因

在`action_provider_sonic.py`的`_run_gear_sonic()`函数中（837-856行），代码虽然计算了机器人状态，但最终使用的是**全0数组**：

```python
# 计算了但没用
motion_joint_pos_step5_ref = gather_temporal_window(...)
motion_joint_vel_step5_ref = gather_temporal_window(...)

# 实际使用的是全0！
motion_joint_pos_step5_full = np.zeros((_STEP5_FRAMES * 29,), dtype=np.float32)
motion_joint_vel_step5_full = np.zeros((_STEP5_FRAMES * 29,), dtype=np.float32)
motion_root_z_step5 = np.zeros((_STEP5_FRAMES,), dtype=np.float32)
# ... 其他字段也都是0
```

Encoder模型训练时，这些字段都有合理的值。当输入全0时，encoder会产生一个固定的latent输出，完全忽略SMPL数据的变化。

### 修复方案

用**当前仿真机器人的实际状态**填充encoder输入的所有字段：

```python
# 使用当前仿真机器人状态（而不是全0）
robot_joint_pos_step5 = gather_temporal_window(
    self._robot_joint_pos_hist, _STEP5_FRAMES, _STEP5_STRIDE
)
robot_joint_vel_step5 = gather_temporal_window(
    self._robot_joint_vel_hist, _STEP5_FRAMES, _STEP5_STRIDE
)

# 填充encoder输入字段（用实际机器人状态）
motion_joint_pos_step5_full = robot_joint_pos_step5.reshape(-1).astype(np.float32)
motion_joint_vel_step5_full = robot_joint_vel_step5.reshape(-1).astype(np.float32)
motion_root_z_step5 = np.full((_STEP5_FRAMES,), 0.75, dtype=np.float32)
motion_root_z = np.array([0.75], dtype=np.float32)
motion_anchor_orient = np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32)
motion_anchor_orient_step5_full = np.tile(motion_anchor_orient, _STEP5_FRAMES)
motion_joint_pos_lowerbody_full = robot_joint_pos_lowerbody.reshape(-1).astype(np.float32)
motion_joint_vel_lowerbody_full = robot_joint_vel_lowerbody.reshape(-1).astype(np.float32)
```

### 验证结果

运行`tools/verify_fix.py`：
- 帧1→2 Latent变化：0.062500 ✅
- 帧2→3 Latent变化：0.062500 ✅

修复成功！Latent现在会随SMPL数据变化而变化。

### 测试步骤

1. 启动pico_server（gear_sonic侧）
2. 启动IsaacLab仿真
3. 在VR中规律甩手（左右甩，约1Hz）
4. 观察仿真中机器人是否跟随VR运动，是否有规律性

### 修改的文件

- `action_provider/action_provider_sonic.py` (837-868行)
  - 用机器人实际状态填充encoder输入
  - 移除过时的诊断代码

### 关键洞察

SONIC encoder是一个**条件编码器**，它需要：
1. SMPL数据（目标姿态）
2. 机器人当前状态（关节位置、速度等）

只提供SMPL数据而其他字段全0，encoder会认为这是一个"无效"或"初始化"状态，产生固定的latent输出。

这类似于条件VAE/GAN，条件信息（机器人状态）和目标信息（SMPL）都必须有合理的值，模型才能正常工作。
