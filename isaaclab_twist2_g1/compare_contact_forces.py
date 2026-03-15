#!/usr/bin/env python3
"""
对比录制和replay的接触力，找出差异根源
"""

import numpy as np
import json
import sys

def analyze_contact_forces(recording_file, debug_json_file):
    """分析接触力差异"""

    print("=" * 80)
    print("接触力和内部状态分析")
    print("=" * 80)
    print()

    # Load recording data
    recording = np.load(recording_file)

    # Check what states are available
    print("录制文件中可用的状态：")
    for key in sorted(recording.keys()):
        if isinstance(recording[key], np.ndarray):
            print(f"  {key}: {recording[key].shape}")
        else:
            print(f"  {key}: {type(recording[key])}")
    print()

    # Load replay debug log
    with open(debug_json_file, 'r') as f:
        debug_data = json.load(f)

    print("=" * 80)
    print("建议：录制额外的内部状态")
    print("=" * 80)
    print()

    print("1. 在录制脚本中添加以下状态的记录：")
    print()
    print("```python")
    print("# 在 action_provider_wh_twist2.py 的 _record_frame_data 中添加：")
    print()
    print("# 接触力")
    print("robot_data['net_contact_forces'] = self.env.scene['robot'].data.net_contact_forces[0].cpu().numpy()")
    print()
    print("# 实际施加的关节力矩")
    print("robot_data['applied_torque'] = self.env.scene['robot'].data.applied_torque[0].cpu().numpy()")
    print()
    print("# 刚体加速度")
    print("robot_data['body_lin_acc_w'] = self.env.scene['robot'].data.body_lin_acc_w[0].cpu().numpy()")
    print("robot_data['body_ang_acc_w'] = self.env.scene['robot'].data.body_ang_acc_w[0].cpu().numpy()")
    print("```")
    print()

    print("2. 在replay脚本的debug logging中添加相同状态的记录")
    print()

    print("3. 对比Frame 0和Frame 1的这些状态，找出差异")
    print()

    print("=" * 80)
    print("关键洞察")
    print("=" * 80)
    print()

    print("即使录制了接触力和力矩，也无法直接'设置'它们，因为：")
    print()
    print("1. 接触力是由接触几何和物理定律计算出来的")
    print("   - 如果位置和速度相同，接触力应该相同")
    print("   - 如果接触力不同，说明接触求解器的状态不同")
    print()
    print("2. 关节力矩是由PD控制器计算出来的")
    print("   - τ = Kp*(q_target - q_current) - Kd*q_vel")
    print("   - 如果输入相同，力矩应该相同")
    print()
    print("3. 如果这些计算结果不同，可能的原因：")
    print("   a) 接触点的初始化状态不同")
    print("   b) 数值精度问题")
    print("   c) 状态设置的时机问题")
    print()

    print("=" * 80)
    print("实验建议")
    print("=" * 80)
    print()

    print("实验1：录制并对比接触力")
    print("  目标：验证接触力是否不同")
    print("  方法：")
    print("    1. 修改录制脚本，添加net_contact_forces记录")
    print("    2. 重新录制数据")
    print("    3. 在replay中对比接触力")
    print("    4. 如果接触力不同，说明接触状态初始化有问题")
    print()

    print("实验2：录制并对比关节力矩")
    print("  目标：验证PD控制器输出是否不同")
    print("  方法：")
    print("    1. 修改录制脚本，添加applied_torque记录")
    print("    2. 重新录制数据")
    print("    3. 在replay中对比力矩")
    print("    4. 如果力矩不同，说明PD控制器计算有问题")
    print()

    print("实验3：逐步对比10个decimation steps")
    print("  目标：找出误差开始累积的位置")
    print("  方法：")
    print("    1. 在每个physics step后记录状态")
    print("    2. 对比录制和replay的每一步")
    print("    3. 找出第一个产生差异的step")
    print()

    print("=" * 80)
    print("最可能的根本原因")
    print("=" * 80)
    print()

    print("基于当前分析，最可能的原因是：")
    print()
    print("【接触状态的初始化差异】")
    print()
    print("录制时：")
    print("  Frame -1 → Frame 0 (自然演化)")
    print("  接触点从上一帧延续，求解器有warm start")
    print()
    print("Replay时：")
    print("  突然设置到Frame 0")
    print("  接触点需要重新检测和初始化")
    print("  求解器从冷启动开始")
    print()
    print("即使位置和速度相同，接触求解器的初始状态不同，")
    print("可能导致接触力略有不同，进而影响整个系统动力学。")
    print()

    print("=" * 80)
    print("解决方案")
    print("=" * 80)
    print()

    print("方案A：接受误差，使用混合模式")
    print("  - 定期重置状态（如每10帧）")
    print("  - 平衡真实性和稳定性")
    print("  - 最实用的方案")
    print()

    print("方案B：尝试改进初始化")
    print("  - 设置初始状态后，运行几个空步骤让系统稳定")
    print("  - 可能减小但无法完全消除误差")
    print()

    print("方案C：使用kinematic replay")
    print("  - 每帧都强制设置状态")
    print("  - 完全匹配录制轨迹")
    print("  - 但不是真正的物理仿真")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_contact_forces.py <recording.npz> <debug.json>")
        print()
        print("Example:")
        print("  python compare_contact_forces.py \\")
        print("    recording_data/Isaac-Move-Football-G129-Dex3-Wholebody_1773326197574686.npz \\")
        print("    replay_debug_logs/Isaac-Move-Football-G129-Dex3-Wholebody_1773326197574686.json")
        sys.exit(1)

    analyze_contact_forces(sys.argv[1], sys.argv[2])
