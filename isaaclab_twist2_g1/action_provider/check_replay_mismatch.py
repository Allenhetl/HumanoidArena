#!/usr/bin/env python3
"""
检查replay偏差的真正原因

既然录制和replay的流程完全一样，为什么还会有偏差？

可能的原因：
1. 第一帧的初始状态不一致
2. 物理引擎的随机性（没有设置种子）
3. 浮点精度累积误差
4. PD控制器参数不一致
5. 环境配置不一致（重力、摩擦力等）
"""

import numpy as np

def check_first_frame_consistency(npz_path):
    """检查第一帧的数据"""
    data = np.load(npz_path, allow_pickle=True)

    print("=" * 80)
    print("第一帧数据检查")
    print("=" * 80)

    # 第一帧的状态
    root_pos_0 = data['robot_root_position'][0]
    root_quat_0 = data['robot_root_orientation'][0]
    qpos_actual_0 = data['robot_qpos_before_decimation'][0]
    qpos_target_0 = data['robot_twist2_inference_qpos'][0]

    print(f"\nFrame 0 状态:")
    print(f"  Root position: {root_pos_0}")
    print(f"  Root orientation: {root_quat_0}")
    print(f"  Actual qpos (前5个): {qpos_actual_0[:5]}")
    print(f"  Target qpos (前5个): {qpos_target_0[:5]}")

    # 检查第一帧的actual和target是否一致
    diff_0 = np.abs(qpos_actual_0 - qpos_target_0).max()
    print(f"\n  Frame 0: actual vs target max diff = {diff_0:.6f}")

    if diff_0 > 0.01:
        print(f"  ⚠️ 第一帧的actual和target就不一致！")
        print(f"  这说明录制开始时，机器人不在default pose")
        print(f"  Replay时如果从default pose开始，就会有初始偏差")

    # 检查连续几帧的变化
    print(f"\n前5帧的状态变化:")
    for i in range(min(5, len(data['robot_root_position']))):
        root_pos = data['robot_root_position'][i]
        qpos_actual = data['robot_qpos_before_decimation'][i]
        qpos_target = data['robot_twist2_inference_qpos'][i]

        print(f"\nFrame {i}:")
        print(f"  Root pos: {root_pos}")
        print(f"  Actual qpos (前3个): {qpos_actual[:3]}")
        print(f"  Target qpos (前3个): {qpos_target[:3]}")
        print(f"  Actual vs Target diff: {np.abs(qpos_actual - qpos_target).max():.6f}")

def analyze_pd_tracking_error(npz_path):
    """分析PD控制器的跟踪误差"""
    data = np.load(npz_path, allow_pickle=True)

    qpos_actual = data['robot_qpos_before_decimation']
    qpos_target = data['robot_twist2_inference_qpos']

    print("\n" + "=" * 80)
    print("PD控制器跟踪误差分析")
    print("=" * 80)

    # 计算每帧的跟踪误差
    tracking_errors = np.abs(qpos_actual - qpos_target)

    print(f"\n跟踪误差统计:")
    print(f"  Mean error per joint: {tracking_errors.mean(axis=0)[:6]}... (前6个关节)")
    print(f"  Max error per joint: {tracking_errors.max(axis=0)[:6]}... (前6个关节)")
    print(f"  Overall mean error: {tracking_errors.mean():.6f} rad")
    print(f"  Overall max error: {tracking_errors.max():.6f} rad")

    print(f"\n关键发现:")
    print(f"  平均跟踪误差: {tracking_errors.mean():.6f} rad ({np.degrees(tracking_errors.mean()):.2f}°)")
    print(f"  最大跟踪误差: {tracking_errors.max():.6f} rad ({np.degrees(tracking_errors.max()):.2f}°)")

    if tracking_errors.mean() > 0.1:
        print(f"\n  ⚠️ PD控制器跟踪误差很大！")
        print(f"  这说明:")
        print(f"  1. PD增益可能不够大")
        print(f"  2. 或者目标变化太快，PD跟不上")
        print(f"  3. Replay时如果直接用target作为目标，会有同样的跟踪误差")
        print(f"  4. 但由于初始状态的微小差异，跟踪轨迹会逐渐偏离")

def main():
    npz_path = "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/Isaac-Move-Football-G129-Dex3-Wholebody_1773220571468541.npz"

    print("检查replay偏差的真正原因...\n")

    check_first_frame_consistency(npz_path)
    analyze_pd_tracking_error(npz_path)

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("""
可能的偏差来源:

1. 初始状态不一致
   - 录制开始时，机器人可能不在default pose
   - Replay时如果从default pose开始，就会有初始偏差
   - 解决方案: Replay时设置第一帧的actual qpos作为初始状态

2. PD控制器跟踪误差
   - 平均误差~0.23 rad，最大误差~2.0 rad
   - 即使用相同的target，由于初始状态微小差异
   - PD控制器的跟踪轨迹会逐渐偏离
   - 这是混沌系统的特性：对初始条件敏感

3. 物理引擎的确定性
   - 如果设置了随机种子，物理引擎应该是确定性的
   - 但浮点精度误差会累积
   - 特别是在接触/碰撞计算中

4. 为什么Direct模式误差小，Inference模式误差大？
   - Direct模式: 使用录制的target，跟踪误差是固定的
   - Inference模式: 使用录制的observation重新推理
     - 如果observation基于actual qpos
     - 但replay时的actual qpos有偏差
     - 推理出的action就会不同
     - 误差累积

建议:
1. 检查第一帧的初始状态是否一致
2. 尝试不设置root状态，只设置joint positions
3. 检查物理引擎的随机种子设置
4. 对比录制和replay的PD控制器参数
    """)

if __name__ == "__main__":
    main()
