#!/usr/bin/env python
"""
计算SONIC Encoder的完整输入维度

根据observation_config.yaml中启用的encoder_observations
"""

# 根据observation_config.yaml第28-56行
encoder_observations = [
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),      # 10 × 29
    ("motion_joint_velocities_10frame_step5", 290),     # 10 × 29
    ("motion_root_z_position_10frame_step5", 10),       # 10 × 1
    ("motion_root_z_position", 1),                      # 1
    ("motion_anchor_orientation", 6),                   # 1 × 6
    ("motion_anchor_orientation_10frame_step5", 60),    # 10 × 6
    ("motion_joint_positions_lowerbody_10frame_step5", 120),  # 10 × 12
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),  # 10 × 12
    ("vr_3point_local_target", 9),                      # 3 × 3
    ("vr_3point_local_orn_target", 12),                 # 3 × 4
    ("smpl_joints_10frame_step1", 720),                 # 10 × 24 × 3
    ("smpl_anchor_orientation_10frame_step1", 60),      # 10 × 6
    ("motion_joint_positions_wrists_10frame_step1", 60), # 10 × 6
]

print("="*80)
print("SONIC Encoder输入维度计算")
print("="*80)

total = 0
for name, dim in encoder_observations:
    print(f"{name:50s} : {dim:4d}")
    total += dim

print("-"*80)
print(f"{'总维度':50s} : {total:4d}")
print("="*80)

print("\n实际encoder输入维度: 1762")
if total == 1762:
    print("✓ 计算匹配！")
else:
    print(f"✗ 不匹配，差异: {1762 - total}")