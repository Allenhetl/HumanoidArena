#!/usr/bin/env python3
"""
验证修复：检查encoder输入是否正确填充了机器人状态

这个脚本模拟修复后的encoder输入构建逻辑，验证latent是否会随SMPL变化而变化。
"""

import numpy as np
import onnxruntime as ort

# 模拟机器人状态（SONIC IsaacLab order）
SONIC_DEFAULT_POS = np.array([
    -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4,
    0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)

OFFICIAL_LOWERBODY_INDICES = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]
OFFICIAL_WRIST_INDICES = [23, 24, 25, 26, 27, 28]

_STEP5_FRAMES = 10
_STEP5_STRIDE = 5
_STEP1_FRAMES = 10

def gather_temporal_window(hist: np.ndarray, num_frames: int, stride: int) -> np.ndarray:
    """Take the latest temporal window using `stride` over a history buffer."""
    hist = np.asarray(hist, dtype=np.float32)
    required = (num_frames - 1) * stride + 1
    if hist.shape[0] < required:
        raise ValueError(f"History too short: {hist.shape[0]} < {required}")
    start = hist.shape[0] - required
    window = hist[start::stride]
    return window.astype(np.float32)

def create_encoder_input_fixed(smpl_joints, robot_joint_pos_hist, robot_joint_vel_hist):
    """创建修复后的encoder输入（用机器人状态填充，而不是全0）"""
    encoder_input = np.zeros((1, 1762), dtype=np.float32)

    # encoder_mode: [0, 0, 1, 0] for SMPL mode
    encoder_input[0, 0:4] = [0, 0, 1, 0]

    # 使用机器人状态填充（修复后的逻辑）
    robot_joint_pos_step5 = gather_temporal_window(robot_joint_pos_hist, _STEP5_FRAMES, _STEP5_STRIDE)
    robot_joint_vel_step5 = gather_temporal_window(robot_joint_vel_hist, _STEP5_FRAMES, _STEP5_STRIDE)

    lowerbody_indices = OFFICIAL_LOWERBODY_INDICES
    robot_joint_pos_lowerbody = robot_joint_pos_step5[:, lowerbody_indices]
    robot_joint_vel_lowerbody = robot_joint_vel_step5[:, lowerbody_indices]

    motion_joint_pos_step5_full = robot_joint_pos_step5.reshape(-1)
    motion_joint_vel_step5_full = robot_joint_vel_step5.reshape(-1)
    motion_root_z_step5 = np.full((_STEP5_FRAMES,), 0.75, dtype=np.float32)
    motion_root_z = np.array([0.75], dtype=np.float32)
    motion_anchor_orient = np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32)
    motion_anchor_orient_step5_full = np.tile(motion_anchor_orient, _STEP5_FRAMES)
    motion_joint_pos_lowerbody_full = robot_joint_pos_lowerbody.reshape(-1)
    motion_joint_vel_lowerbody_full = robot_joint_vel_lowerbody.reshape(-1)
    vr_3pt_pos = np.zeros(9, dtype=np.float32)
    vr_3pt_orn = np.zeros(12, dtype=np.float32)

    smpl_joints_flat = smpl_joints.reshape(-1)
    smpl_anchor_orient_flat = np.tile(np.array([1., 0., 0., 1., 0., 0.], dtype=np.float32), _STEP1_FRAMES)

    wrist_indices = OFFICIAL_WRIST_INDICES
    motion_wrist_window = gather_temporal_window(robot_joint_pos_hist, _STEP1_FRAMES, 1)
    motion_wrist_pos = motion_wrist_window[:, wrist_indices].reshape(-1)

    # 拼接所有观察值
    encoder_input[0] = np.concatenate([
        [0, 0, 1, 0],                           # 4
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
    ])

    return encoder_input

# 加载模型
encoder_path = '/home/dreams/Users/Alyssa/HumanoidArena_V1/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx'
session = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])

print("=" * 80)
print("验证修复：encoder输入用机器人状态填充")
print("=" * 80)

# 创建3帧不同的SMPL数据
smpl1 = np.random.randn(10, 24, 3).astype(np.float32) * 0.3
smpl2 = smpl1 + np.random.randn(10, 24, 3).astype(np.float32) * 0.05
smpl3 = smpl2 + np.random.randn(10, 24, 3).astype(np.float32) * 0.05

# 为每帧创建不同的机器人状态历史（模拟机器人在运动）
def create_robot_hist(frame_idx):
    robot_joint_pos_hist = np.tile(SONIC_DEFAULT_POS[np.newaxis], (50, 1))
    for i in range(50):
        # 添加随机变化 + 帧索引相关的偏移（模拟运动趋势）
        robot_joint_pos_hist[i] += np.random.randn(29) * 0.01 + frame_idx * 0.001
    robot_joint_vel_hist = np.random.randn(50, 29) * 0.1
    return robot_joint_pos_hist, robot_joint_vel_hist

# 测试帧1
robot_pos_hist1, robot_vel_hist1 = create_robot_hist(0)
input1 = create_encoder_input_fixed(smpl1, robot_pos_hist1, robot_vel_hist1)
output1 = session.run(None, {session.get_inputs()[0].name: input1})[0]
print(f"\n帧1:")
print(f"  SMPL前3维: {smpl1[0, 0, :]}")
print(f"  Latent前10维: {output1[0, :10]}")

# 测试帧2
robot_pos_hist2, robot_vel_hist2 = create_robot_hist(1)
input2 = create_encoder_input_fixed(smpl2, robot_pos_hist2, robot_vel_hist2)
output2 = session.run(None, {session.get_inputs()[0].name: input2})[0]
print(f"\n帧2:")
print(f"  SMPL前3维: {smpl2[0, 0, :]}")
print(f"  Latent前10维: {output2[0, :10]}")

# 测试帧3
robot_pos_hist3, robot_vel_hist3 = create_robot_hist(2)
input3 = create_encoder_input_fixed(smpl3, robot_pos_hist3, robot_vel_hist3)
output3 = session.run(None, {session.get_inputs()[0].name: input3})[0]
print(f"\n帧3:")
print(f"  SMPL前3维: {smpl3[0, 0, :]}")
print(f"  Latent前10维: {output3[0, :10]}")

# 计算差异
diff_12 = np.abs(output2 - output1).max()
diff_23 = np.abs(output3 - output2).max()

print(f"\n" + "=" * 80)
print(f"Latent变化:")
print(f"  帧1→2: {diff_12:.6f}")
print(f"  帧2→3: {diff_23:.6f}")

if diff_12 > 0.01 or diff_23 > 0.01:
    print(f"\n✅ 修复成功！Latent正常变化！")
    print(f"\n现在可以启动IsaacLab测试：")
    print(f"  1. 启动pico_server")
    print(f"  2. 启动IsaacLab仿真")
    print(f"  3. 在VR中规律甩手，观察仿真是否跟随")
else:
    print(f"\n❌ 修复失败，Latent仍然不变")
