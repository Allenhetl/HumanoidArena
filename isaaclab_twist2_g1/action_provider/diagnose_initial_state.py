#!/usr/bin/env python3
"""
诊断初始状态设置问题
检查Frame 0的初始化是否正确
"""

import numpy as np

# 加载录制数据
recording_file = '/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/Isaac-Move-Football-G129-Dex3-Wholebody_1773326197574686.npz'
data = np.load(recording_file, allow_pickle=True)

print("=" * 80)
print("初始状态诊断")
print("=" * 80)

# Frame 0 数据（用于初始化）
print("\n【Frame 0 - 初始化数据】")
print("-" * 80)

if 'robot_qpos_before_decimation' in data:
    qpos_actual_0 = data['robot_qpos_before_decimation'][0]
    print(f"关节实际位置 (qpos_actual[0]):")
    print(f"  前5个: {qpos_actual_0[:5]}")
    print(f"  全部29个: {qpos_actual_0}")
else:
    print("⚠️ 缺少 robot_qpos_before_decimation")
    qpos_actual_0 = None

if 'robot_qvel_before_decimation' in data:
    qvel_0 = data['robot_qvel_before_decimation'][0]
    print(f"\n关节速度 (qvel[0]):")
    print(f"  前5个: {qvel_0[:5]}")
    print(f"  L2 norm: {np.linalg.norm(qvel_0):.4f}")
else:
    print("⚠️ 缺少 robot_qvel_before_decimation")
    qvel_0 = None

if 'robot_root_position' in data:
    root_pos_0 = data['robot_root_position'][0]
    print(f"\nRoot位置: {root_pos_0}")
else:
    print("⚠️ 缺少 robot_root_position")

if 'robot_root_orientation' in data:
    root_quat_0 = data['robot_root_orientation'][0]
    print(f"Root姿态 (w,x,y,z): {root_quat_0}")
else:
    print("⚠️ 缺少 robot_root_orientation")

if 'robot_root_lin_vel_world' in data:
    root_lin_vel_0 = data['robot_root_lin_vel_world'][0]
    print(f"Root线速度 (world): {root_lin_vel_0}")
elif 'robot_root_lin_vel_local' in data:
    root_lin_vel_0 = data['robot_root_lin_vel_local'][0]
    print(f"Root线速度 (local): {root_lin_vel_0}")
else:
    print("⚠️ 缺少 root_lin_vel")

if 'robot_root_ang_vel_world' in data:
    root_ang_vel_0 = data['robot_root_ang_vel_world'][0]
    print(f"Root角速度 (world): {root_ang_vel_0}")
elif 'robot_root_ang_vel_local' in data:
    root_ang_vel_0 = data['robot_root_ang_vel_local'][0]
    print(f"Root角速度 (local): {root_ang_vel_0}")
else:
    print("⚠️ 缺少 root_ang_vel")

# Frame 1 数据（第一个执行的动作和期望的结果）
print("\n\n【Frame 1 - 第一个动作和期望结果】")
print("-" * 80)

if 'robot_twist2_inference_qpos' in data:
    qpos_1 = data['robot_twist2_inference_qpos'][1]
    print(f"动作 (qpos[1] - PD目标):")
    print(f"  前5个: {qpos_1[:5]}")
else:
    print("⚠️ 缺少 robot_twist2_inference_qpos")
    qpos_1 = None

if qpos_actual_0 is not None:
    qpos_actual_1 = data['robot_qpos_before_decimation'][1]
    print(f"\n期望结果 (qpos_actual[1]):")
    print(f"  前5个: {qpos_actual_1[:5]}")

    if qpos_1 is not None:
        print(f"\n对比分析:")
        print(f"  Frame 0实际位置 → Frame 1目标 → Frame 1实际位置")
        for i in range(5):
            delta_target = qpos_1[i] - qpos_actual_0[i]
            delta_actual = qpos_actual_1[i] - qpos_actual_0[i]
            print(f"  Joint {i}: {qpos_actual_0[i]:7.4f} → (目标{qpos_1[i]:7.4f}, Δ{delta_target:7.4f}) → {qpos_actual_1[i]:7.4f} (实际Δ{delta_actual:7.4f})")

# Debug日志中的Frame 1仿真结果
print("\n\n【Debug日志中的Frame 1仿真结果】")
print("-" * 80)
simulated_1 = np.array([0.118165, -0.085819, 0.067047, -0.018324, -0.050896])
print(f"仿真结果 (前5个): {simulated_1}")

if qpos_actual_0 is not None:
    qpos_actual_1 = data['robot_qpos_before_decimation'][1]
    print(f"\n误差分析:")
    print(f"  录制 vs 仿真:")
    for i in range(5):
        error = qpos_actual_1[i] - simulated_1[i]
        print(f"  Joint {i}: 录制={qpos_actual_1[i]:7.4f}, 仿真={simulated_1[i]:7.4f}, 误差={error:7.4f}")

    l2_error = np.linalg.norm(qpos_actual_1[:5] - simulated_1)
    print(f"\n  L2误差 (前5个关节): {l2_error:.4f} rad")

# 可能的问题
print("\n\n【可能的问题】")
print("-" * 80)
print("1. 初始状态设置后是否立即生效？")
print("   → 检查是否需要运行physics step让状态稳定")
print("\n2. 速度是否正确设置？")
print("   → 检查速度坐标系（world vs local）")
print("\n3. Decimation循环是否正确？")
print("   → 检查physics step数量和时间步长")
print("\n4. 随机种子是否完全确定性？")
print("   → 检查CUDA、cuDNN、PhysX的随机性")
print("\n5. 初始化时机是否正确？")
print("   → 检查是否在env.reset()之后立即设置")

print("\n" + "=" * 80)
