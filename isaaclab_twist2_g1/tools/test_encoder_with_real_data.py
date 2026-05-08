#!/usr/bin/env python3
"""
使用实际的encoder输入测试模型

从日志中提取实际的encoder输入，然后用encoder模型推理，
看看输出是否真的不变。
"""

import numpy as np
import onnxruntime as ort

# 模拟实际的encoder输入（基于日志）
# 前10维: [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]  <- encoder_mode等
# 922-932维: SMPL数据（会变化）

def create_encoder_input_with_smpl(smpl_values):
    """创建1762维encoder输入，只有SMPL部分不同"""
    encoder_input = np.zeros((1, 1762), dtype=np.float32)

    # encoder_mode: [0, 0, 1, 0]
    encoder_input[0, 0:4] = [0, 0, 1, 0]

    # SMPL joints在922-1642位置（720维）
    # 这里我们只设置前10维作为测试
    encoder_input[0, 922:932] = smpl_values

    # 其他部分保持0（模拟实际情况）
    return encoder_input

# 加载模型
encoder_path = '/home/dreams/Users/Alyssa/HumanoidArena_V1/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx'
session = ort.InferenceSession(encoder_path, providers=['CPUExecutionProvider'])

print("=" * 80)
print("测试：用实际的SMPL数据测试encoder模型")
print("=" * 80)

# 从日志中提取的实际SMPL数据（前10维）
smpl_frame1 = np.array([-0.33130527, -0.11529671, 0.02412217, -0.35730642, -0.05710728,
                         -0.06864123, -0.33644223, -0.10098453, 0.01558272, -0.35244334], dtype=np.float32)
smpl_frame2 = np.array([-0.32548583, -0.13167374, 0.01900305, -0.35148698, -0.07348433,
                         -0.07376036, -0.34496915, -0.06361398, 0.0242058, -0.37097025], dtype=np.float32)
smpl_frame3 = np.array([-0.31801605, -0.14822504, 0.02316006, -0.3440171, -0.09003565,
                         -0.06960335, -0.3427359, -0.0722274, 0.03093787, -0.368737], dtype=np.float32)

# 测试1
input1 = create_encoder_input_with_smpl(smpl_frame1)
output1 = session.run(None, {session.get_inputs()[0].name: input1})[0]
print(f"\n帧1:")
print(f"  SMPL输入前3维: {smpl_frame1[:3]}")
print(f"  Latent输出前10维: {output1[0, :10]}")
print(f"  Latent范围: [{output1.min():.6f}, {output1.max():.6f}]")

# 测试2
input2 = create_encoder_input_with_smpl(smpl_frame2)
output2 = session.run(None, {session.get_inputs()[0].name: input2})[0]
print(f"\n帧2:")
print(f"  SMPL输入前3维: {smpl_frame2[:3]}")
print(f"  Latent输出前10维: {output2[0, :10]}")
print(f"  Latent范围: [{output2.min():.6f}, {output2.max():.6f}]")

# 测试3
input3 = create_encoder_input_with_smpl(smpl_frame3)
output3 = session.run(None, {session.get_inputs()[0].name: input3})[0]
print(f"\n帧3:")
print(f"  SMPL输入前3维: {smpl_frame3[:3]}")
print(f"  Latent输出前10维: {output3[0, :10]}")
print(f"  Latent范围: [{output3.min():.6f}, {output3.max():.6f}]")

# 计算差异
diff_12 = np.abs(output2 - output1).max()
diff_23 = np.abs(output3 - output2).max()

print(f"\n" + "=" * 80)
print(f"Latent变化:")
print(f"  帧1→2: {diff_12:.6f}")
print(f"  帧2→3: {diff_23:.6f}")

if diff_12 < 1e-6 and diff_23 < 1e-6:
    print(f"\n❌ 问题确认：即使SMPL输入变化，Latent输出也不变！")
    print(f"   可能原因：")
    print(f"   1. 模型训练时SMPL数据没有被使用")
    print(f"   2. 模型期望的输入格式不同")
    print(f"   3. 需要填充其他字段（不能全是0）")
else:
    print(f"\n✅ Latent正常变化")
    print(f"   说明问题不在模型，而在IsaacLab的encoder输入构建")
