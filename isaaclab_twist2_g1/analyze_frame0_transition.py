#!/usr/bin/env python3
"""分析 Frame 0 到 Frame 1 的转换"""

import numpy as np

# 加载录制数据
data = np.load('recording_data/Isaac-Move-Football-G129-Dex3-Wholebody_1773326197574686.npz')

print("="*80)
print("Frame 0 → Frame 1 转换分析")
print("="*80)

print("\n1. Frame 0 初始状态:")
print(f"   Joint Position (前5个): {data['robot_qpos_before_decimation'][0][:5]}")
print(f"   Joint Velocity (前5个): {data['robot_qvel_before_decimation'][0][:5]}")
print(f"   Joint Velocity L2 norm: {np.linalg.norm(data['robot_qvel_before_decimation'][0]):.4f} rad/s")

print("\n2. Frame 0 应用的 Action (ONNX output):")
print(f"   Action (前5个): {data['robot_twist2_inference_qpos'][0][:5]}")

print("\n3. Frame 1 期望状态 (应用 action[0] 后):")
print(f"   Joint Position (前5个): {data['robot_qpos_before_decimation'][1][:5]}")
print(f"   Joint Velocity (前5个): {data['robot_qvel_before_decimation'][1][:5]}")
print(f"   Joint Velocity L2 norm: {np.linalg.norm(data['robot_qvel_before_decimation'][1]):.4f} rad/s")

print("\n4. 位置变化 (Frame 1 - Frame 0):")
pos_change = data['robot_qpos_before_decimation'][1] - data['robot_qpos_before_decimation'][0]
print(f"   Delta Position (前5个): {pos_change[:5]}")
print(f"   Delta Position L2 norm: {np.linalg.norm(pos_change):.4f} rad")

print("\n5. 速度变化 (Frame 1 - Frame 0):")
vel_change = data['robot_qvel_before_decimation'][1] - data['robot_qvel_before_decimation'][0]
print(f"   Delta Velocity (前5个): {vel_change[:5]}")
print(f"   Delta Velocity L2 norm: {np.linalg.norm(vel_change):.4f} rad/s")

print("\n6. Action 与 Frame 0 Position 的差异:")
action_diff = data['robot_twist2_inference_qpos'][0] - data['robot_qpos_before_decimation'][0]
print(f"   Action - Pos0 (前5个): {action_diff[:5]}")
print(f"   Action - Pos0 L2 norm: {np.linalg.norm(action_diff):.4f} rad")

print("\n7. 关键观察:")
print(f"   - Frame 0 的速度已经很大 ({np.linalg.norm(data['robot_qvel_before_decimation'][0]):.2f} rad/s)")
print(f"   - 应用 action[0] 后，速度变化了 {np.linalg.norm(vel_change):.2f} rad/s")
print(f"   - 这表明 PD 控制器在调整速度以跟踪目标位置")

print("\n8. 潜在问题:")
print("   如果 replay 中的 PD 控制器参数（stiffness, damping）与录制时不同，")
print("   那么即使应用相同的 action，产生的速度变化也会不同。")

print("\n9. 建议:")
print("   - 检查 PD 控制器参数是否一致")
print("   - 检查 decimation 步数是否一致 (应该是 10)")
print("   - 检查 physics_dt 是否一致")
