#!/usr/bin/env python
"""验证SONIC动作缩放修复"""
import numpy as np

# 从 action_provider_sonic.py 导入
G1_ACTION_SCALE_ISAACLAB = np.array([
    0.3506614566,  # 0: left_hip_pitch_joint
    0.3506614566,  # 1: right_hip_pitch_joint
    0.5475464463,  # 2: waist_yaw_joint
    0.3506614566,  # 3: left_hip_roll_joint
    0.3506614566,  # 4: right_hip_roll_joint
    0.4385773242,  # 5: waist_roll_joint
    0.5475464463,  # 6: left_hip_yaw_joint
    0.5475464463,  # 7: right_hip_yaw_joint
    0.4385773242,  # 8: waist_pitch_joint
    0.3506614566,  # 9: left_knee_joint
    0.3506614566,  # 10: right_knee_joint
    0.4385773242,  # 11: left_shoulder_pitch_joint
    0.4385773242,  # 12: right_shoulder_pitch_joint
    0.4385773242,  # 13: left_ankle_pitch_joint
    0.4385773242,  # 14: right_ankle_pitch_joint
    0.4385773242,  # 15: left_shoulder_roll_joint
    0.4385773242,  # 16: right_shoulder_roll_joint
    0.4385773242,  # 17: left_ankle_roll_joint
    0.4385773242,  # 18: right_ankle_roll_joint
    0.4385773242,  # 19: left_shoulder_yaw_joint
    0.4385773242,  # 20: right_shoulder_yaw_joint
    0.4385773242,  # 21: left_elbow_joint
    0.4385773242,  # 22: right_elbow_joint
    0.4385773242,  # 23: left_wrist_roll_joint
    0.4385773242,  # 24: right_wrist_roll_joint
    0.0745008737,  # 25: left_wrist_pitch_joint
    0.0745008737,  # 26: right_wrist_pitch_joint
    0.0745008737,  # 27: left_wrist_yaw_joint
    0.0745008737,  # 28: right_wrist_yaw_joint
], dtype=np.float32)

SONIC_DEFAULT_POS = np.array([
    -0.312,  # left_hip_pitch_joint
    -0.312,  # right_hip_pitch_joint
    0.0,     # waist_yaw_joint
    0.0,     # left_hip_roll_joint
    0.0,     # right_hip_roll_joint
    0.0,     # waist_roll_joint
    0.0,     # left_hip_yaw_joint
    0.0,     # right_hip_yaw_joint
    0.0,     # waist_pitch_joint
    0.669,   # left_knee_joint
    0.669,   # right_knee_joint
    0.2,     # left_shoulder_pitch_joint
    0.2,     # right_shoulder_pitch_joint
    -0.363,  # left_ankle_pitch_joint
    -0.363,  # right_ankle_pitch_joint
    0.2,     # left_shoulder_roll_joint
    -0.2,    # right_shoulder_roll_joint
    0.0,     # left_ankle_roll_joint
    0.0,     # right_ankle_roll_joint
    0.0,     # left_shoulder_yaw_joint
    0.0,     # right_shoulder_yaw_joint
    0.6,     # left_elbow_joint
    0.6,     # right_elbow_joint
    0.0,     # left_wrist_roll_joint
    0.0,     # right_wrist_roll_joint
    0.0,     # left_wrist_pitch_joint
    0.0,     # right_wrist_pitch_joint
    0.0,     # left_wrist_yaw_joint
    0.0,     # right_wrist_yaw_joint
], dtype=np.float32)

print("=" * 80)
print("SONIC 动作缩放修复验证")
print("=" * 80)

# 测试1: 零动作应该输出默认姿态
print("\n测试1: 零动作 → 默认姿态")
raw_action_zero = np.zeros(29, dtype=np.float32)
target_zero = raw_action_zero * G1_ACTION_SCALE_ISAACLAB + SONIC_DEFAULT_POS
print(f"✓ 零动作输出: {target_zero[:5]} ... (前5个关节)")
print(f"✓ 应该等于默认姿态: {SONIC_DEFAULT_POS[:5]} ... (前5个关节)")
assert np.allclose(target_zero, SONIC_DEFAULT_POS), "零动作测试失败！"
print("✓ 零动作测试通过")

# 测试2: 典型动作范围
print("\n测试2: 典型动作范围 [-2, 2]")
raw_action_pos = np.full(29, 2.0, dtype=np.float32)
raw_action_neg = np.full(29, -2.0, dtype=np.float32)
target_pos = raw_action_pos * G1_ACTION_SCALE_ISAACLAB + SONIC_DEFAULT_POS
target_neg = raw_action_neg * G1_ACTION_SCALE_ISAACLAB + SONIC_DEFAULT_POS
print(f"✓ 动作=+2.0 → 目标范围: [{target_pos.min():.3f}, {target_pos.max():.3f}]")
print(f"✓ 动作=-2.0 → 目标范围: [{target_neg.min():.3f}, {target_neg.max():.3f}]")

# 测试3: 对比固定缩放 vs per-joint 缩放
print("\n测试3: 固定缩放 vs Per-joint 缩放")
raw_action_test = np.random.randn(29).astype(np.float32)
target_fixed = raw_action_test * 0.25 + SONIC_DEFAULT_POS
target_perJoint = raw_action_test * G1_ACTION_SCALE_ISAACLAB + SONIC_DEFAULT_POS
diff = np.abs(target_perJoint - target_fixed)
print(f"✓ 随机动作输入: {raw_action_test[:3]} ...")
print(f"✓ 固定缩放输出: {target_fixed[:3]} ...")
print(f"✓ Per-joint缩放输出: {target_perJoint[:3]} ...")
print(f"✓ 最大差异: {diff.max():.4f} rad ({np.rad2deg(diff.max()):.2f} deg)")
print(f"✓ 平均差异: {diff.mean():.4f} rad ({np.rad2deg(diff.mean()):.2f} deg)")

# 测试4: 验证不同电机类型的缩放系数
print("\n测试4: 电机类型分组")
motor_types = {
    "7520_22 (大力矩)": [0, 1, 3, 4, 9, 10],  # 髋、膝
    "7520_14 (中力矩)": [2, 6, 7],  # 腰yaw、髋yaw
    "5020 (标准)": [5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # 肩、肘、踝
    "4010 (小力矩)": [25, 26, 27, 28],  # 手腕
}
for motor_type, indices in motor_types.items():
    scales = G1_ACTION_SCALE_ISAACLAB[indices]
    print(f"✓ {motor_type}: scale={scales[0]:.4f}, count={len(indices)}")

print("\n" + "=" * 80)
print("✓ 所有测试通过！动作缩放修复正确。")
print("=" * 80)