#!/usr/bin/env python3
"""
分析真正的replay误差来源

关键问题：
1. 如果observation完全相同，为什么inference结果会不同？
2. 如果设置了随机种子，为什么物理状态会有偏差？
3. Direct模式和Inference模式的target_29应该完全相同吗？
"""

import numpy as np

def analyze_recorded_data(npz_path):
    """分析录制数据，找出真正的误差来源"""
    data = np.load(npz_path, allow_pickle=True)

    print("=" * 80)
    print("录制数据分析")
    print("=" * 80)

    # 1. 检查录制的qpos和observation的一致性
    qpos = data['robot_twist2_inference_qpos']  # [N, 29] - 录制时的target_29
    obs_buf = data['robot_obs_buf']  # [N, 1432]

    print(f"\n1. 数据维度检查:")
    print(f"   qpos shape: {qpos.shape}")
    print(f"   obs_buf shape: {obs_buf.shape}")

    # 2. 检查qpos是否包含在observation中
    # Observation结构: [obs_full(127) + obs_hist(1270) + future(35)]
    # obs_full = [action_mimic(35) + obs_proprio(92)]
    # obs_proprio = [ang_vel(3) + roll_pitch(2) + dof_pos_delta(29) + dof_vel(29) + last_action(29)]

    print(f"\n2. Observation结构分析:")
    print(f"   obs_full (0:127): 当前观测")
    print(f"   obs_hist (127:1397): 历史观测 (10帧 * 127)")
    print(f"   future_obs (1397:1432): 未来mimic (35)")

    # 提取last_action (在obs_proprio的最后29维)
    # obs_proprio在obs_full的35:127位置
    # last_action在obs_proprio的63:92位置 (3+2+29+29 = 63开始)
    last_action_in_obs = obs_buf[:, 35+63:35+92]  # [N, 29]

    print(f"\n3. 检查observation中的last_action:")
    print(f"   last_action_in_obs shape: {last_action_in_obs.shape}")
    print(f"   Frame 0 last_action: {last_action_in_obs[0, :5]}... (前5个)")
    print(f"   Frame 1 last_action: {last_action_in_obs[1, :5]}... (前5个)")

    # 4. 检查qpos和last_action的关系
    # 注意：Frame N的qpos应该等于Frame N+1的last_action (因为last_action是上一帧的输出)
    print(f"\n4. 检查qpos和last_action的时序关系:")
    for i in range(min(3, len(qpos)-1)):
        # qpos[i]是Frame i的输出，应该出现在Frame i+1的observation中
        diff = np.abs(qpos[i] - last_action_in_obs[i+1]).max()
        print(f"   Frame {i}: qpos vs Frame {i+1} last_action, max diff = {diff:.6f}")

    # 5. 检查录制时的物理状态
    root_pos = data['robot_root_position']
    root_quat = data['robot_root_orientation']
    qpos_before_decimation = data['robot_qpos_before_decimation']

    print(f"\n5. 物理状态数据:")
    print(f"   root_position shape: {root_pos.shape}")
    print(f"   root_orientation shape: {root_quat.shape}")
    print(f"   qpos_before_decimation shape: {qpos_before_decimation.shape}")

    # 6. 关键问题：qpos_before_decimation vs twist2_inference_qpos
    print(f"\n6. 检查qpos_before_decimation vs twist2_inference_qpos:")
    for i in range(min(3, len(qpos))):
        diff = np.abs(qpos_before_decimation[i] - qpos[i]).max()
        print(f"   Frame {i}: max diff = {diff:.6f}")
        if diff > 0.01:
            print(f"   ⚠️ 差异较大！qpos_before_decimation是物理仿真后的实际值")
            print(f"   twist2_inference_qpos是模型输出的目标值")

    # 7. 关键发现：录制时保存的是什么？
    print(f"\n7. 录制时序分析:")
    print(f"   录制时保存的数据:")
    print(f"   - robot_twist2_inference_qpos: 模型输出的target_29 (目标值)")
    print(f"   - robot_qpos_before_decimation: 物理仿真后的实际关节位置")
    print(f"   - robot_root_position/orientation: 物理仿真后的实际root状态")
    print(f"   - robot_obs_buf: 基于实际物理状态构建的observation")

    print(f"\n8. Replay时的问题:")
    print(f"   Direct模式:")
    print(f"   - 使用twist2_inference_qpos (目标值)")
    print(f"   - 但录制时的observation是基于qpos_before_decimation (实际值)")
    print(f"   - 如果目标值≠实际值，就会有不一致！")
    print(f"")
    print(f"   Inference模式:")
    print(f"   - 使用录制的observation")
    print(f"   - 推理出新的target_29")
    print(f"   - 如果observation是基于实际值，但我们用目标值replay，就会偏离")

    # 9. 验证：检查目标值和实际值的差异
    print(f"\n9. 目标值 vs 实际值的差异统计:")
    diffs = np.abs(qpos_before_decimation - qpos)
    print(f"   Mean diff per joint: {diffs.mean(axis=0)[:6]}... (前6个关节)")
    print(f"   Max diff per joint: {diffs.max(axis=0)[:6]}... (前6个关节)")
    print(f"   Overall mean diff: {diffs.mean():.6f}")
    print(f"   Overall max diff: {diffs.max():.6f}")

    if diffs.mean() > 0.01:
        print(f"\n   ⚠️ 发现显著差异！这说明:")
        print(f"   1. PD控制器没有完美跟踪目标位置")
        print(f"   2. 录制的observation基于实际位置，不是目标位置")
        print(f"   3. Replay时如果只用目标位置，会与observation不匹配")

    return data

def main():
    npz_path = "/home/dreams/Users/taowen/HumanoidArena/isaaclab_twist2_g1/recording_data/Isaac-Move-Football-G129-Dex3-Wholebody_1773220571468541.npz"

    print("分析录制数据以找出真正的误差来源...\n")
    data = analyze_recorded_data(npz_path)

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("\n关键发现:")
    print("1. 录制时保存了两种qpos:")
    print("   - twist2_inference_qpos: 模型输出的目标位置")
    print("   - qpos_before_decimation: 物理仿真后的实际位置")
    print("")
    print("2. Observation是基于实际位置构建的，不是目标位置")
    print("")
    print("3. Replay时的不一致:")
    print("   - Direct模式: 使用目标位置，但observation基于实际位置")
    print("   - Inference模式: 使用录制的observation (基于实际位置)，推理出新目标")
    print("")
    print("4. 解决方案:")
    print("   - Direct模式: 应该使用qpos_before_decimation而不是twist2_inference_qpos")
    print("   - Inference模式: 应该使用录制的observation (已经正确)")
    print("   - 都需要设置root状态为录制的实际状态")

if __name__ == "__main__":
    main()
