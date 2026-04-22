#!/usr/bin/env python3
"""
测试：用合理的非零值填充encoder输入

假设：模型训练时，即使在SMPL模式下，其他字段也有合理的值（比如当前机器人状态）
"""

import numpy as np
import onnxruntime as ort

def create_realistic_encoder_input(smpl_joints, frame_idx=0):
    """创建1762维encoder输入，用合理的值填充所有字段"""
    encoder_input = np.zeros((1, 1762), dtype=np.float32)

    # encoder_mode: [0, 0, 1, 0] for SMPL mode
    encoder_input[0, 0:4] = [0, 0, 1, 0]

    # motion_joint_positions_10frame_step5: 290维
    # 用合理的关节位置填充（模拟机器人当前姿态）
    # G1有29个关节，10帧
    joint_pos = np.array([
        -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4,
        0.0, 0.0, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ], dtype=np.float32)
    # 添加一些随机变化模拟运动
    for i in range(10):
        offset = 4 + i * 29
        encoder_input[0, offset:offset+29] = joint_pos + np.random.randn(29) * 0.01

    # motion_joint_velocities_10frame_step5: 290维
    # 用小的随机值模拟速度
    encoder_input[0, 294:584] = np.random.randn(290) * 0.1

    # motion_root_z_position_10frame_step5: 10维
    encoder_input[0, 584:594] = 0.75 + np.random.randn(10) * 0.01  # 机器人高度约0.75m

    # motion_root_z_position: 1维
    encoder_input[0, 594] = 0.75

    # motion_anchor_orientation: 6维 (6D rotation)
    encoder_input[0, 595:601] = [1, 0, 0, 0, 1, 0]  # 单位旋转

    # motion_anchor_orientation_10frame_step5: 60维
    for i in range(10):
        encoder_input[0, 601+i*6:601+(i+1)*6] = [1, 0, 0, 0, 1, 0]

    # motion_joint_positions_lowerbody_10frame_step5: 120维 (12个下半身关节 x 10帧)
    lowerbody_pos = joint_pos[:12]
    for i in range(10):
        encoder_input[0, 661+i*12:661+(i+1)*12] = lowerbody_pos + np.random.randn(12) * 0.01

    # motion_joint_velocities_lowerbody_10frame_step5: 120维
    encoder_input[0, 781:901] = np.random.randn(120) * 0.1

    # vr_3point_local_target: 9维 (3点 x 3坐标)
    encoder_input[0, 901:910] = np.random.randn(9) * 0.1

    # vr_3point_local_orn_target: 12维 (3点 x 4四元数)
    for i in range(3):
        encoder_input[0, 910+i*4:910+(i+1)*4] = [1, 0, 0, 0]  # 单位四元数

    # smpl_joints_10frame_step1: 720维 (24关节 x 3坐标 x 10帧)
    # 这是关键数据！
    encoder_input[0, 922:922+720] = smpl_joints

    # smpl_anchor_orientation_10frame_step1: 60维
    for i in range(10):
        encoder_input[0, 1642+i*6:1642+(i+1)*6] = [1, 0, 0, 0, 1, 0]

    # motion_joint_positions_wrists_10frame_step1: 60维 (6个手腕关节 x 10帧)
    wrist_pos = joint_pos[23:29]  # 假设23-28是手腕关节
    for i in range(10):
        encoder_input[0, 1702+i*6:1702+(i+1)*6] = wrist_pos + np.random.randn(6) * 0.01

    return encoder_input

# 加载模型
encoder_path = '/home/dreams/Users/Alyssa/HumanoidArena_V1/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx'
session = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])

print("=" * 80)
print("测试：用合理的非零值填充encoder输入")
print("=" * 80)

# 创建3帧不同的SMPL数据
smpl1 = np.random.randn(720) * 0.3
smpl2 = smpl1 + np.random.randn(720) * 0.05  # 小幅变化
smpl3 = smpl2 + np.random.randn(720) * 0.05

# 测试
input1 = create_realistic_encoder_input(smpl1, 0)
output1 = session.run(None, {session.get_inputs()[0].name: input1})[0]
print(f"\n帧1:")
print(f"  SMPL前3维: {smpl1[:3]}")
print(f"  Latent前10维: {output1[0, :10]}")

input2 = create_realistic_encoder_input(smpl2, 1)
output2 = session.run(None, {session.get_inputs()[0].name: input2})[0]
print(f"\n帧2:")
print(f"  SMPL前3维: {smpl2[:3]}")
print(f"  Latent前10维: {output2[0, :10]}")

input3 = create_realistic_encoder_input(smpl3, 2)
output3 = session.run(None, {session.get_inputs()[0].name: input3})[0]
print(f"\n帧3:")
print(f"  SMPL前3维: {smpl3[:3]}")
print(f"  Latent前10维: {output3[0, :10]}")

# 计算差异
diff_12 = np.abs(output2 - output1).max()
diff_23 = np.abs(output3 - output2).max()

print(f"\n" + "=" * 80)
print(f"Latent变化:")
print(f"  帧1→2: {diff_12:.6f}")
print(f"  帧2→3: {diff_23:.6f}")

if diff_12 > 0.01 or diff_23 > 0.01:
    print(f"\n✅ 成功！用合理的非零值填充后，Latent正常变化！")
    print(f"\n解决方案：")
    print(f"  在IsaacLab中，需要用机器人当前状态填充encoder输入的其他字段")
    print(f"  不能让motion_joint_positions等字段全是0")
else:
    print(f"\n❌ 即使填充合理值，Latent仍然不变")
    print(f"   可能需要检查模型训练配置或输入格式")
