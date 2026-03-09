#!/usr/bin/env python
"""计算SONIC IsaacLab order的action scales"""
import numpy as np

# Motor constants from policy_parameters.hpp
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2

# Stiffness values
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ

# Effort limits
EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0

# g1_action_scale in MuJoCo order (from policy_parameters.hpp lines 109-139)
g1_action_scale_mujoco = np.array([
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 0: left_hip_pitch_joint
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 1: left_hip_roll_joint
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # 2: left_hip_yaw_joint
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 3: left_knee_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 4: left_ankle_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 5: left_ankle_roll_joint
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 6: right_hip_pitch_joint
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 7: right_hip_roll_joint
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # 8: right_hip_yaw_joint
    0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 9: right_knee_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 10: right_ankle_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 11: right_ankle_roll_joint
    0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # 12: waist_yaw_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 13: waist_roll_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 14: waist_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 15: left_shoulder_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 16: left_shoulder_roll_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 17: left_shoulder_yaw_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 18: left_elbow_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 19: left_wrist_roll_joint
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,        # 20: left_wrist_pitch_joint
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,        # 21: left_wrist_yaw_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 22: right_shoulder_pitch_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 23: right_shoulder_roll_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 24: right_shoulder_yaw_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 25: right_elbow_joint
    0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 26: right_wrist_roll_joint
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,        # 27: right_wrist_pitch_joint
    0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,        # 28: right_wrist_yaw_joint
], dtype=np.float32)

# MuJoCo to IsaacLab mapping (from policy_parameters.hpp line 103)
mujoco_to_isaaclab = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                                16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])

# Convert to IsaacLab order
g1_action_scale_isaaclab = g1_action_scale_mujoco[mujoco_to_isaaclab]

# SONIC IsaacLab joint order (from action_provider_sonic.py)
SONIC_ISAACLAB_JOINT_ORDER = [
    "left_hip_pitch_joint",      # 0
    "right_hip_pitch_joint",     # 1
    "waist_yaw_joint",           # 2
    "left_hip_roll_joint",       # 3
    "right_hip_roll_joint",      # 4
    "waist_roll_joint",          # 5
    "left_hip_yaw_joint",        # 6
    "right_hip_yaw_joint",       # 7
    "waist_pitch_joint",         # 8
    "left_knee_joint",           # 9
    "right_knee_joint",          # 10
    "left_shoulder_pitch_joint", # 11
    "right_shoulder_pitch_joint",# 12
    "left_ankle_pitch_joint",    # 13
    "right_ankle_pitch_joint",   # 14
    "left_shoulder_roll_joint",  # 15
    "right_shoulder_roll_joint", # 16
    "left_ankle_roll_joint",     # 17
    "right_ankle_roll_joint",    # 18
    "left_shoulder_yaw_joint",   # 19
    "right_shoulder_yaw_joint",  # 20
    "left_elbow_joint",          # 21
    "right_elbow_joint",         # 22
    "left_wrist_roll_joint",     # 23
    "right_wrist_roll_joint",    # 24
    "left_wrist_pitch_joint",    # 25
    "right_wrist_pitch_joint",   # 26
    "left_wrist_yaw_joint",      # 27
    "right_wrist_yaw_joint",     # 28
]

print("=" * 80)
print("SONIC Action Scales (IsaacLab order)")
print("=" * 80)
print("\nPython array for action_provider_sonic.py:\n")
print("G1_ACTION_SCALE_ISAACLAB = np.array([")
for i, (name, scale) in enumerate(zip(SONIC_ISAACLAB_JOINT_ORDER, g1_action_scale_isaaclab)):
    print(f"    {scale:.10f},  # {i}: {name}")
print("], dtype=np.float32)")

print("\n" + "=" * 80)
print("Verification:")
print("=" * 80)
print(f"Min scale: {g1_action_scale_isaaclab.min():.6f}")
print(f"Max scale: {g1_action_scale_isaaclab.max():.6f}")
print(f"Mean scale: {g1_action_scale_isaaclab.mean():.6f}")
print(f"\nUnique values: {len(np.unique(g1_action_scale_isaaclab))}")
print(f"Shape: {g1_action_scale_isaaclab.shape}")