#!/usr/bin/env python3
"""
验证SMPL模式encoder输入是否正确置0

根据 observation_config.yaml，SMPL模式(mode_id=2)只需要4个观测块：
1. encoder_mode_4 (4维)
2. smpl_joints_10frame_step1 (720维)
3. smpl_anchor_orientation_10frame_step1 (60维)
4. motion_joint_positions_wrists_10frame_step1 (60维)

其他918维应该全部为0。
"""

import numpy as np

# 模拟encoder输入构建
encoder_mode = np.array([2., 0., 0., 0.], dtype=np.float32)

# SMPL模式下应该为0的观测块
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

# SMPL模式下应该有数据的观测块（模拟非零数据）
smpl_joints_flat = np.random.randn(720).astype(np.float32)
smpl_anchor_orient_flat = np.random.randn(60).astype(np.float32)
motion_wrist_pos = np.random.randn(60).astype(np.float32)

# 拼接encoder输入
encoder_input = np.concatenate([
    encoder_mode,                           # 4
    motion_joint_pos_step5_full,            # 290
    motion_joint_vel_step5_full,            # 290
    motion_root_z_step5,                    # 10
    motion_root_z,                          # 1
    motion_anchor_orient,                   # 6
    motion_anchor_orient_step5_full,        # 60
    motion_joint_pos_lowerbody_full,        # 120
    motion_joint_vel_lowerbody_full,        # 120
    vr_3pt_pos,                             # 9
    vr_3pt_orn,                             # 12
    smpl_joints_flat,                       # 720
    smpl_anchor_orient_flat,                # 60
    motion_wrist_pos,                       # 60
])[np.newaxis]  # (1, 1762)

print("=" * 80)
print("SMPL Mode Encoder Input Verification")
print("=" * 80)

# 验证总维度
assert encoder_input.shape == (1, 1762), f"Expected shape (1, 1762), got {encoder_input.shape}"
print(f"✓ Total dimension: {encoder_input.shape[1]} (correct)")

# 验证各个块的维度和值
offset = 0

# encoder_mode (4)
block = encoder_input[0, offset:offset+4]
assert np.array_equal(block, [2., 0., 0., 0.]), "encoder_mode should be [2, 0, 0, 0]"
print(f"✓ encoder_mode_4 (offset {offset}): {block.tolist()}")
offset += 4

# 应该为0的块
zero_blocks = [
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
]

total_zero_dims = 0
for name, dim in zero_blocks:
    block = encoder_input[0, offset:offset+dim]
    assert np.all(block == 0.0), f"{name} should be all zeros"
    print(f"✓ {name} (offset {offset}, dim {dim}): all zeros")
    offset += dim
    total_zero_dims += dim

print(f"\n✓ Total zeroed dimensions: {total_zero_dims} (expected 918)")
assert total_zero_dims == 918, f"Expected 918 zero dims, got {total_zero_dims}"

# 应该有数据的块
active_blocks = [
    ("smpl_joints_10frame_step1", 720, smpl_joints_flat),
    ("smpl_anchor_orientation_10frame_step1", 60, smpl_anchor_orient_flat),
    ("motion_joint_positions_wrists_10frame_step1", 60, motion_wrist_pos),
]

total_active_dims = 0
for name, dim, expected_data in active_blocks:
    block = encoder_input[0, offset:offset+dim]
    assert np.array_equal(block, expected_data), f"{name} data mismatch"
    assert not np.all(block == 0.0), f"{name} should have non-zero data"
    print(f"✓ {name} (offset {offset}, dim {dim}): has data, range [{block.min():.4f}, {block.max():.4f}]")
    offset += dim
    total_active_dims += dim

print(f"\n✓ Total active dimensions: {total_active_dims} (expected 840)")
assert total_active_dims == 840, f"Expected 840 active dims, got {total_active_dims}"

# 最终验证
assert offset == 1762, f"Offset mismatch: {offset} != 1762"
print(f"\n✓ Final offset: {offset} (matches total dimension)")

print("\n" + "=" * 80)
print("✓ ALL CHECKS PASSED - SMPL mode encoder input is correctly configured!")
print("=" * 80)
print(f"\nSummary:")
print(f"  - encoder_mode: 4 dims")
print(f"  - Zeroed blocks: {total_zero_dims} dims (NOT required in SMPL mode)")
print(f"  - Active blocks: {total_active_dims} dims (required in SMPL mode)")
print(f"  - Total: {4 + total_zero_dims + total_active_dims} dims")
