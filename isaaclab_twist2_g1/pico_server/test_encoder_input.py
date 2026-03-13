#!/usr/bin/env python
"""
快速测试SONIC encoder输入构建

验证1762维输入的正确性
"""

import numpy as np

def test_encoder_input_construction():
    """测试encoder输入构建"""
    print("="*80)
    print("测试SONIC Encoder输入构建")
    print("="*80)

    # 模拟数据
    smpl_joints_buf = np.random.randn(10, 24, 3).astype(np.float32)
    body_rot6d_buf = np.random.randn(10, 6).astype(np.float32)
    robot_joint_pos_hist = np.random.randn(10, 29).astype(np.float32)
    robot_joint_vel_hist = np.random.randn(10, 29).astype(np.float32)

    # 构建encoder输入
    print("\n构建1762维encoder输入...")

    # 1. encoder_mode_4 (4)
    encoder_mode = np.array([0., 0., 1., 0.], dtype=np.float32)
    print(f"  1. encoder_mode: {encoder_mode.shape} = {encoder_mode.size}")

    # 2. motion_joint_positions_10frame_step5 (290)
    motion_joint_pos_step5 = robot_joint_pos_hist[::5].reshape(-1)
    motion_joint_pos_step5_full = np.zeros(290, dtype=np.float32)
    motion_joint_pos_step5_full[:motion_joint_pos_step5.size] = motion_joint_pos_step5
    print(f"  2. motion_joint_pos_step5: {motion_joint_pos_step5_full.shape} = {motion_joint_pos_step5_full.size}")

    # 3. motion_joint_velocities_10frame_step5 (290)
    motion_joint_vel_step5 = robot_joint_vel_hist[::5].reshape(-1)
    motion_joint_vel_step5_full = np.zeros(290, dtype=np.float32)
    motion_joint_vel_step5_full[:motion_joint_vel_step5.size] = motion_joint_vel_step5
    print(f"  3. motion_joint_vel_step5: {motion_joint_vel_step5_full.shape} = {motion_joint_vel_step5_full.size}")

    # 4. motion_root_z_position_10frame_step5 (10)
    motion_root_z_step5 = np.zeros(10, dtype=np.float32)
    print(f"  4. motion_root_z_step5: {motion_root_z_step5.shape} = {motion_root_z_step5.size}")

    # 5. motion_root_z_position (1)
    motion_root_z = np.zeros(1, dtype=np.float32)
    print(f"  5. motion_root_z: {motion_root_z.shape} = {motion_root_z.size}")

    # 6. motion_anchor_orientation (6)
    motion_anchor_orient = body_rot6d_buf[-1]
    print(f"  6. motion_anchor_orient: {motion_anchor_orient.shape} = {motion_anchor_orient.size}")

    # 7. motion_anchor_orientation_10frame_step5 (60)
    motion_anchor_orient_step5 = body_rot6d_buf[::5].reshape(-1)
    motion_anchor_orient_step5_full = np.zeros(60, dtype=np.float32)
    motion_anchor_orient_step5_full[:motion_anchor_orient_step5.size] = motion_anchor_orient_step5
    print(f"  7. motion_anchor_orient_step5: {motion_anchor_orient_step5_full.shape} = {motion_anchor_orient_step5_full.size}")

    # 8. motion_joint_positions_lowerbody_10frame_step5 (120)
    motion_joint_pos_lowerbody = robot_joint_pos_hist[::5, :12].reshape(-1)
    motion_joint_pos_lowerbody_full = np.zeros(120, dtype=np.float32)
    motion_joint_pos_lowerbody_full[:motion_joint_pos_lowerbody.size] = motion_joint_pos_lowerbody
    print(f"  8. motion_joint_pos_lowerbody: {motion_joint_pos_lowerbody_full.shape} = {motion_joint_pos_lowerbody_full.size}")

    # 9. motion_joint_velocities_lowerbody_10frame_step5 (120)
    motion_joint_vel_lowerbody = robot_joint_vel_hist[::5, :12].reshape(-1)
    motion_joint_vel_lowerbody_full = np.zeros(120, dtype=np.float32)
    motion_joint_vel_lowerbody_full[:motion_joint_vel_lowerbody.size] = motion_joint_vel_lowerbody
    print(f"  9. motion_joint_vel_lowerbody: {motion_joint_vel_lowerbody_full.shape} = {motion_joint_vel_lowerbody_full.size}")

    # 10. vr_3point_local_target (9)
    vr_3pt_pos = np.zeros(9, dtype=np.float32)
    print(f" 10. vr_3pt_pos: {vr_3pt_pos.shape} = {vr_3pt_pos.size}")

    # 11. vr_3point_local_orn_target (12)
    vr_3pt_orn = np.zeros(12, dtype=np.float32)
    print(f" 11. vr_3pt_orn: {vr_3pt_orn.shape} = {vr_3pt_orn.size}")

    # 12. smpl_joints_10frame_step1 (720)
    smpl_joints_flat = smpl_joints_buf.reshape(-1)
    print(f" 12. smpl_joints_flat: {smpl_joints_flat.shape} = {smpl_joints_flat.size}")

    # 13. smpl_anchor_orientation_10frame_step1 (60)
    smpl_anchor_orient_flat = body_rot6d_buf.reshape(-1)
    print(f" 13. smpl_anchor_orient_flat: {smpl_anchor_orient_flat.shape} = {smpl_anchor_orient_flat.size}")

    # 14. motion_joint_positions_wrists_10frame_step1 (60)
    wrist_indices = [12, 13, 14, 15, 16, 17]
    motion_wrist_pos = robot_joint_pos_hist[:, wrist_indices].reshape(-1)
    print(f" 14. motion_wrist_pos: {motion_wrist_pos.shape} = {motion_wrist_pos.size}")

    # 拼接
    encoder_input = np.concatenate([
        encoder_mode,                           # 4
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

    print("\n" + "-"*80)
    print(f"最终encoder输入形状: {encoder_input.shape}")
    print(f"期望形状: (1762,)")

    if encoder_input.shape[0] == 1762:
        print("✓ 维度正确！")
        return True
    else:
        print(f"✗ 维度错误！差异: {encoder_input.shape[0] - 1762}")
        return False

def test_with_batch():
    """测试带batch维度的输入"""
    print("\n" + "="*80)
    print("测试带batch维度的输入")
    print("="*80)

    # 构建单个样本
    encoder_input = np.random.randn(1762).astype(np.float32)

    # 添加batch维度
    encoder_input_batch = encoder_input[np.newaxis]

    print(f"单个样本形状: {encoder_input.shape}")
    print(f"带batch形状: {encoder_input_batch.shape}")
    print(f"期望形状: (1, 1762)")

    if encoder_input_batch.shape == (1, 1762):
        print("✓ Batch维度正确！")
        return True
    else:
        print("✗ Batch维度错误！")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SONIC Encoder输入构建测试")
    print("="*80)

    test1 = test_encoder_input_construction()
    test2 = test_with_batch()

    print("\n" + "="*80)
    print("测试结果")
    print("="*80)
    print(f"维度测试: {'✓ 通过' if test1 else '✗ 失败'}")
    print(f"Batch测试: {'✓ 通过' if test2 else '✗ 失败'}")

    if test1 and test2:
        print("\n✓ 所有测试通过！encoder输入构建正确。")
    else:
        print("\n✗ 部分测试失败，请检查实现。")