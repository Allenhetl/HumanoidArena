#!/usr/bin/env python3
"""
分析录制和replay的时序差异

关键问题：录制时的数据采集时机导致时序不匹配
"""

import numpy as np

def analyze_timing_mismatch(npz_path):
    """分析时序不匹配问题"""
    data = np.load(npz_path, allow_pickle=True)

    print("=" * 80)
    print("时序分析")
    print("=" * 80)

    # 录制时的流程
    print("\n录制时的流程 (action_provider_wh_twist2.py):")
    print("=" * 80)
    print("""
Frame N (get_action开始):
  ├─ 1. compute_observations()
  │    └─ 读取当前状态 (这是Frame N-1仿真后的结果！)
  │        ├─ root_state_w
  │        ├─ joint_pos
  │        └─ joint_vel
  │
  ├─ 2. run_policy(obs)
  │    └─ ONNX推理 → raw_action → target_29
  │
  ├─ 3. collect_recording_data(obs_buf, target_29)  ← 在物理仿真之前！
  │    └─ 保存:
  │        ├─ robot_root_position: Frame N-1仿真后的位置
  │        ├─ robot_root_orientation: Frame N-1仿真后的朝向
  │        ├─ robot_qpos_before_decimation: Frame N-1仿真后的关节位置
  │        ├─ robot_twist2_inference_qpos: Frame N的target_29
  │        └─ robot_obs_buf: 基于Frame N-1状态的observation
  │
  └─ 4. 物理仿真 (decimation=10步)
       ├─ set_joint_position_target(target_29)
       ├─ for i in range(10):
       │    ├─ write_data_to_sim()
       │    ├─ sim.step()
       │    └─ scene.update()
       └─ 结果: Frame N仿真后的新状态
    """)

    print("\nReplay时的流程 (当前实现):")
    print("=" * 80)
    print("""
Frame N (get_action开始):
  ├─ 1. 从录制数据加载
  │    ├─ root_pos = recorded_root_position[N]  ← Frame N-1仿真后的状态
  │    ├─ root_quat = recorded_root_orientation[N]
  │    └─ target_29 = recorded_twist2_inference_qpos[N]  ← Frame N的目标
  │
  ├─ 2. 设置root状态
  │    └─ write_root_pose_to_sim(root_pos, root_quat)
  │        ⚠️ 问题：我们设置的是Frame N-1的状态
  │        但马上要用Frame N的target_29进行仿真！
  │
  └─ 3. 物理仿真 (decimation=10步)
       ├─ set_joint_position_target(target_29)
       └─ for i in range(10):
            ├─ write_data_to_sim()
            ├─ sim.step()
            └─ scene.update()
    """)

    print("\n时序不匹配的影响:")
    print("=" * 80)
    print("""
录制时:
  Frame N-1状态 → Frame N observation → Frame N target_29 → 仿真 → Frame N状态

Replay时:
  设置Frame N-1状态 → 用Frame N target_29仿真 → 得到不同的Frame N状态

问题:
  1. 录制时，Frame N的target_29是基于Frame N-1状态的observation推理出来的
  2. Replay时，我们设置Frame N-1状态，然后用Frame N的target_29
  3. 但是！Frame N-1状态可能与录制时的Frame N-1状态略有不同
  4. 这个微小差异会导致物理仿真结果不同
  5. 误差逐帧累积
    """)

    # 检查连续帧之间的状态变化
    root_pos = data['robot_root_position']
    root_quat = data['robot_root_orientation']
    qpos_actual = data['robot_qpos_before_decimation']
    qpos_target = data['robot_twist2_inference_qpos']

    print("\n连续帧状态变化分析:")
    print("=" * 80)

    # 计算Frame N的target和Frame N-1的actual的关系
    print("\n检查: Frame N的target是否基于Frame N-1的actual状态?")
    for i in range(1, min(5, len(qpos_target))):
        # Frame i的target应该是基于Frame i-1的actual状态推理出来的
        pos_change = np.linalg.norm(root_pos[i] - root_pos[i-1])
        qpos_change = np.linalg.norm(qpos_actual[i] - qpos_actual[i-1])
        target_change = np.linalg.norm(qpos_target[i] - qpos_target[i-1])

        print(f"\nFrame {i-1} → Frame {i}:")
        print(f"  Root position change: {pos_change:.6f} m")
        print(f"  Actual qpos change: {qpos_change:.6f} rad")
        print(f"  Target qpos change: {target_change:.6f} rad")

        # 检查target和actual的差异
        target_actual_diff = np.linalg.norm(qpos_target[i] - qpos_actual[i-1])
        print(f"  Frame {i} target vs Frame {i-1} actual: {target_actual_diff:.6f} rad")

    print("\n" + "=" * 80)
    print("关键发现:")
    print("=" * 80)
    print("""
1. 录制时保存的root状态是"上一帧仿真后"的状态
2. 录制时保存的target_29是"当前帧ONNX输出"的目标
3. 这两者之间有时序差异！

4. Replay时的问题:
   - 我们设置"上一帧的状态"
   - 然后用"当前帧的目标"进行仿真
   - 但由于累积误差，"上一帧的状态"可能已经不准确
   - 导致仿真结果偏离

5. 为什么Inference模式误差更大?
   - Inference模式使用录制的observation
   - 但observation是基于"上一帧的实际状态"
   - 如果replay时的"上一帧状态"不准确
   - observation就不准确
   - 推理出的action就不准确
   - 误差指数级累积
    """)

    print("\n解决方案:")
    print("=" * 80)
    print("""
方案1: 不设置root状态 (让物理引擎自然演化)
  优点: 物理连续性最好
  缺点: 轨迹会偏离录制轨迹

方案2: 同时保存root速度，replay时设置速度而不是位置
  优点: 更平滑，物理上更合理
  缺点: 需要修改录制脚本

方案3: 接受误差，使用周期性校正
  优点: 简单，不需要修改录制脚本
  缺点: 会有周期性的小跳变

方案4: 修改录制脚本，保存"仿真后"的状态
  优点: 时序匹配
  缺点: 需要重新录制数据
    """)

def main():
    npz_path = "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/Isaac-Move-Football-G129-Dex3-Wholebody_1773220571468541.npz"

    print("分析录制和replay的时序差异...\n")
    analyze_timing_mismatch(npz_path)

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("""
核心问题: 时序不匹配

录制时:
  保存的root状态 = Frame N-1仿真后
  保存的target_29 = Frame N ONNX输出
  保存的observation = 基于Frame N-1状态

Replay时:
  设置Frame N-1状态 + 用Frame N target_29仿真
  → 如果Frame N-1状态有误差，仿真结果就会偏离
  → 误差累积

这就是为什么即使使用相同的observation和target，
replay结果仍然会偏离录制轨迹的根本原因！

最佳实践:
1. Direct模式: 不设置root状态，让物理自然演化
2. Inference模式: 使用录制的observation，但不强制root状态
3. 如果需要精确复现: 修改录制脚本，保存"仿真后"的完整状态
    """)

if __name__ == "__main__":
    main()
