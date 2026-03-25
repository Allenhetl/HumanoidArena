# SMPL Mode Encoder Input Fix

## 问题描述

Python实现的 `action_provider_sonic.py` 与C++原始实现 `g1_deploy_onnx_ref.cpp` 在SMPL模式下的encoder输入存在重大差异。

### 原始问题

根据 `observation_config.yaml:74-80`，SMPL模式（mode_id=2）只需要4个观测块：

1. `encoder_mode_4` (4维)
2. `smpl_joints_10frame_step1` (720维)
3. `smpl_anchor_orientation_10frame_step1` (60维)
4. `motion_joint_positions_wrists_10frame_step1` (60维)

**其他918维应该全部为0**，但Python实现中这些观测块都有数据，导致：
- 训练-推理输入分布不匹配
- 模型行为不可预测
- 可能的性能下降

## 修复内容

### 修改文件
- `action_provider_sonic.py:1455-1479`

### 修改前（错误实现）

```python
# ✨ CRITICAL FIX: Use reference motion data from ZMQ instead of zeros
# In SMPL mode, the encoder still needs robot proprioception for better tracking
motion_joint_pos_step5_full = motion_joint_pos_step5_ref.reshape(-1).astype(np.float32)
motion_joint_vel_step5_full = motion_joint_vel_step5_ref.reshape(-1).astype(np.float32)
motion_root_z_step5 = gather_temporal_window(...).reshape(-1).astype(np.float32)
motion_root_z = np.array([self._motion_root_z_hist[-1]], dtype=np.float32)
motion_anchor_orient = self._body_rot6d_buf[-1].copy()
motion_anchor_orient_step5_full = gather_temporal_window(...).reshape(-1).astype(np.float32)
motion_joint_pos_lowerbody_full = motion_joint_pos_lowerbody_ref.reshape(-1).astype(np.float32)
motion_joint_vel_lowerbody_full = motion_joint_vel_lowerbody_ref.reshape(-1).astype(np.float32)
vr_3pt_pos = self._vr_3pt_position.copy()
vr_3pt_orn = self._vr_3pt_orientation.copy()
```

**问题**：所有这些观测块都有数据，与C++实现不一致。

### 修改后（正确实现）

```python
# ============================================================================
# SMPL Mode (mode_id=2) Encoder Input Construction
# ============================================================================
# According to observation_config.yaml, SMPL mode only requires 4 observation blocks:
#   1. encoder_mode_4
#   2. smpl_joints_10frame_step1
#   3. smpl_anchor_orientation_10frame_step1
#   4. motion_joint_positions_wrists_10frame_step1
#
# All other observations must be ZERO to match C++ implementation.
# Reference: gear_sonic_deploy/policy/release/observation_config.yaml:74-80
# Reference: g1_deploy_onnx_ref.cpp:1920-1942 (mode filtering logic)
# ============================================================================

# These observations are NOT required in SMPL mode → set to ZERO
motion_joint_pos_step5_full = np.zeros(290, dtype=np.float32)
motion_joint_vel_step5_full = np.zeros(290, dtype=np.float32)
motion_root_z_step5 = np.zeros(10, dtype=np.float32)
motion_root_z = np.zeros(1, dtype=np.float32)
motion_anchor_orient = np.zeros(6, dtype=np.float32)
motion_anchor_orient_step5_full = np.zeros(60, dtype=np.float32)
motion_joint_pos_lowerbody_full = np.zeros(120, dtype=np.float32)
motion_joint_vel_lowerbody_full = np.zeros(120, dtype=np.float32)
vr_3pt_pos = np.zeros(9, dtype=np.float32)
vr_3pt_orn = np.zeros(12, dtype=np.float32)
```

## 验证

运行验证脚本：
```bash
python verify_smpl_mode_fix.py
```

验证结果：
```
✓ Total dimension: 1762 (correct)
✓ encoder_mode_4: [2.0, 0.0, 0.0, 0.0]
✓ Zeroed blocks: 918 dims (NOT required in SMPL mode)
✓ Active blocks: 840 dims (required in SMPL mode)
✓ ALL CHECKS PASSED
```

## Encoder输入结构（SMPL模式）

| Offset | Dimension | Block Name | Value | Required? |
|--------|-----------|------------|-------|-----------|
| 0 | 4 | encoder_mode_4 | [2, 0, 0, 0] | ✅ Yes |
| 4 | 290 | motion_joint_positions_10frame_step5 | **0** | ❌ No |
| 294 | 290 | motion_joint_velocities_10frame_step5 | **0** | ❌ No |
| 584 | 10 | motion_root_z_position_10frame_step5 | **0** | ❌ No |
| 594 | 1 | motion_root_z_position | **0** | ❌ No |
| 595 | 6 | motion_anchor_orientation | **0** | ❌ No |
| 601 | 60 | motion_anchor_orientation_10frame_step5 | **0** | ❌ No |
| 661 | 120 | motion_joint_positions_lowerbody_10frame_step5 | **0** | ❌ No |
| 781 | 120 | motion_joint_velocities_lowerbody_10frame_step5 | **0** | ❌ No |
| 901 | 9 | vr_3point_local_target | **0** | ❌ No |
| 910 | 12 | vr_3point_local_orn_target | **0** | ❌ No |
| 922 | 720 | smpl_joints_10frame_step1 | **SMPL data** | ✅ Yes |
| 1642 | 60 | smpl_anchor_orientation_10frame_step1 | **SMPL data** | ✅ Yes |
| 1702 | 60 | motion_joint_positions_wrists_10frame_step1 | **Wrist data** | ✅ Yes |
| **1762** | **Total** | | | |

## 参考

### C++ 原始实现
- **配置文件**：`/home/dreams/Users/taowen/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml:74-80`
- **模式过滤逻辑**：`g1_deploy_onnx_ref.cpp:1920-1942`
  - 代码注释："Not required for this mode - leave as zero"

### Python 修复实现
- **修改文件**：`action_provider_sonic.py:1455-1479`
- **验证脚本**：`verify_smpl_mode_fix.py`

## 预期效果

修复后，Python实现的encoder输入将与C++原始实现完全一致：
- ✅ 训练-推理输入分布匹配
- ✅ 模型行为可预测
- ✅ 性能稳定
- ✅ 与官方部署代码行为一致

## 测试建议

1. 运行验证脚本确认修改正确
2. 在仿真环境中测试SMPL模式遥操作
3. 对比修复前后的机器人行为
4. 检查encoder latent输出的稳定性
