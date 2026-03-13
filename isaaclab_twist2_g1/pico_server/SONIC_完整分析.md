# SONIC推理问题 - 完整分析与修复方案

## 问题回顾

用户报告：`sonic_targets` 始终输出固定的默认值，SONIC模型推理不工作。

## 根本原因

**数据格式错误**: IsaacLab发送四元数(4维)，但SONIC encoder期望6D旋转表示(6维)

## 已完成的修复

### 1. 添加四元数到6D旋转转换函数 ✓

在 `action_provider_sonic.py` 中添加了 `quat_to_rotation_6d()` 函数：

```python
def quat_to_rotation_6d(quat: np.ndarray) -> np.ndarray:
    """将四元数转换为6D旋转表示（旋转矩阵前2列）"""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # 旋转矩阵前2列
    r00 = 1 - 2*(y*y + z*z)
    r10 = 2*(x*y + w*z)
    r20 = 2*(x*z - w*y)

    r01 = 2*(x*y - w*z)
    r11 = 1 - 2*(x*x + z*z)
    r21 = 2*(y*z + w*x)

    rot6d = np.stack([r00, r01, r10, r11, r20, r21], axis=-1)
    return rot6d.astype(np.float32)
```

### 2. 修改缓冲区从四元数改为6D旋转 ✓

```python
# 修改前: (10, 4)
self._body_quat_buf = np.tile(
    np.array([1., 0., 0., 0.], dtype=np.float32), (10, 1))

# 修改后: (10, 6)
self._body_rot6d_buf = np.tile(
    np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32), (10, 1))
```

### 3. 修改ZMQ数据接收，添加转换 ✓

```python
if "body_quat_w" in data:
    bq = data["body_quat_w"].astype(np.float32)  # (N, 4)
    # 转换为6D旋转表示
    rot6d = quat_to_rotation_6d(bq)  # (N, 6)
    self._body_rot6d_buf = np.roll(self._body_rot6d_buf, -1, axis=0)
    self._body_rot6d_buf[-1] = rot6d[-1]
```

### 4. 修改Encoder输入使用6D旋转 ✓

```python
anchor_orient = self._body_rot6d_buf[np.newaxis]  # (1, 10, 6)
```

## 需要进一步验证的问题

### 关键问题：Encoder输入格式

根据官方文档 (`Observation Configuration — GR00T-WholeBodyControl Documentation.pdf`)，SMPL模式的encoder需要：

1. **encoder_mode_4** (4维) - 模式标识 `[0, 0, 1, 0]`
2. **smpl_joints_10frame_step1** (720维) - 10帧×24关节×3坐标
3. **smpl_anchor_orientation_10frame_step1** (60维) - 10帧×6
4. **motion_joint_positions_wrists_10frame_step1** (60维) - 10帧×6个手腕关节

**总维度**: 4 + 720 + 60 + 60 = **844维**

### 当前实现的潜在问题

当前代码传递的是：
```python
enc_inputs = {
    self._encoder.get_inputs()[0].name: smpl_joints_in,  # (1, 10, 24, 3)
    self._encoder.get_inputs()[1].name: anchor_orient,   # (1, 10, 6)
    self._encoder.get_inputs()[2].name: joint_pos_hist,  # (1, 10, 29)
}
```

**可能的问题**:
1. 缺少 `encoder_mode_4` 输入
2. 数据可能需要展平为1D向量
3. `joint_pos_hist` 应该是手腕关节(6个)，不是全部关节(29个)

### 两种可能的Encoder输入格式

#### 格式A: 单输入，所有特征拼接

```python
encoder_input = np.concatenate([
    np.array([0., 0., 1., 0.]),                    # encoder_mode (4,)
    self._smpl_joints_buf.reshape(-1),             # smpl_joints (720,)
    self._body_rot6d_buf.reshape(-1),              # anchor_orient (60,)
    wrist_pos_hist.reshape(-1),                    # wrist_pos (60,)
])[np.newaxis]  # (1, 844)

enc_inputs = {
    self._encoder.get_inputs()[0].name: encoder_input
}
```

#### 格式B: 多输入，分别传递

```python
enc_inputs = {
    'encoder_mode': np.array([[0., 0., 1., 0.]]),           # (1, 4)
    'smpl_joints': self._smpl_joints_buf.reshape(1, -1),    # (1, 720)
    'anchor_orient': self._body_rot6d_buf.reshape(1, -1),   # (1, 60)
    'wrist_pos': wrist_pos_hist.reshape(1, -1),             # (1, 60)
}
```

#### 格式C: 多维输入（当前实现）

```python
enc_inputs = {
    'smpl_joints': smpl_joints_in,    # (1, 10, 24, 3)
    'anchor_orient': anchor_orient,   # (1, 10, 6)
    'joint_pos': joint_pos_hist,      # (1, 10, 29) 或 (1, 10, 6)
}
```

## 验证步骤

### 步骤1: 检查Encoder模型输入

运行检查脚本：
```bash
python isaaclab_twist2_g1/pico_server/check_sonic_encoder.py /path/to/model_encoder.onnx
```

这将显示：
- Encoder有几个输入
- 每个输入的名称和形状
- 是否需要展平

### 步骤2: 根据实际输入调整代码

根据步骤1的结果，修改 `action_provider_sonic.py` 中的 `_run_gear_sonic()` 方法。

### 步骤3: 测试转换函数

```bash
python isaaclab_twist2_g1/pico_server/test_quat_to_rot6d.py
```

确保四元数到6D旋转的转换正确。

### 步骤4: 运行完整系统

```bash
# Terminal 1
python isaaclab_twist2_g1/pico_server/pico_server_pose_only.py --vis_vr3pt

# Terminal 2
bash isaaclab_twist2_g1/run_sonic.sh
```

观察日志输出，检查：
- Encoder输入形状是否正确
- 是否有推理错误
- sonic_targets是否变化

## 参考文档

1. **官方文档**: `/Users/taowenwang/Downloads/Observation Configuration — GR00T-WholeBodyControl Documentation.pdf`
2. **观察配置**: `gear_sonic_deploy/policy/release/observation_config.yaml`
3. **C++实现**: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`
4. **在线文档**: `gear_sonic_deploy/docs/source/references/observation_config.md`

## 关键发现

1. ✓ **6D旋转表示**: 确认encoder使用6D旋转（旋转矩阵前2列），不是四元数
2. ✓ **维度**: `smpl_anchor_orientation_10frame_step1` 是60维 (10×6)，不是40维 (10×4)
3. ✓ **转换函数**: 已实现并与C++代码一致
4. ? **输入格式**: 需要验证是单输入拼接还是多输入分离
5. ? **encoder_mode**: 需要确认是否需要显式传递模式标识
6. ? **手腕关节**: 需要确认是否需要单独的手腕关节历史

## 下一步行动

1. **立即**: 运行 `check_sonic_encoder.py` 检查模型输入格式
2. **根据结果**: 调整 `_run_gear_sonic()` 中的encoder输入构建
3. **测试**: 运行完整系统验证修复效果
4. **优化**: 移除调试打印，优化性能

## 总结

核心修复（四元数→6D旋转）已完成，这解决了最根本的数据格式错误。但还需要验证encoder的具体输入格式（单输入vs多输入，是否需要展平，是否需要encoder_mode等），才能确保完全正确。

建议用户先运行 `check_sonic_encoder.py` 查看实际的encoder输入要求，然后根据结果进行最终调整。