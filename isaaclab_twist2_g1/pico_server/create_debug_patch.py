#!/usr/bin/env python
"""
SONIC 推理问题诊断补丁

在 action_provider_sonic.py 的关键位置添加调试输出
"""

import sys

def create_debug_patch():
    """生成调试补丁代码"""

    patch_code = '''
# ============================================================================
# 调试补丁 - 添加到 _run_gear_sonic() 函数开头
# ============================================================================

def _run_gear_sonic_debug(self) -> np.ndarray:
    """带调试输出的 GEAR-SONIC 推理"""

    print("\\n" + "="*70)
    print("SONIC 推理调试信息")
    print("="*70)

    # 1. 检查模型是否加载
    print(f"\\n1. 模型加载状态:")
    print(f"   Encoder: {self._encoder is not None}")
    print(f"   Decoder: {self._decoder is not None}")

    if self._encoder is None or self._decoder is None:
        print("   ✗ 模型未加载，返回默认姿态")
        return self._sonic_default_np.copy()

    # 2. 检查SMPL数据有效性
    print(f"\\n2. SMPL数据有效性:")
    print(f"   _smpl_data_valid: {self._smpl_data_valid}")
    print(f"   _smpl_joints_buf shape: {self._smpl_joints_buf.shape}")
    print(f"   _smpl_joints_buf 最新帧:")
    print(f"   {self._smpl_joints_buf[-1]}")
    print(f"   绝对值和: {np.abs(self._smpl_joints_buf[-1]).sum():.4f}")

    if not self._smpl_data_valid:
        print("   ✗ SMPL数据无效，返回默认姿态")
        return self._sonic_default_np.copy()

    # 3. 准备Encoder输入
    print(f"\\n3. Encoder输入:")
    smpl_joints_in = self._smpl_joints_buf[np.newaxis]
    anchor_orient  = self._body_quat_buf[np.newaxis]
    joint_pos_hist = np.tile(
        self._robot_joint_pos[np.newaxis, np.newaxis],
        (1, 10, 1)).astype(np.float32)

    print(f"   smpl_joints_in shape: {smpl_joints_in.shape}")
    print(f"   anchor_orient shape: {anchor_orient.shape}")
    print(f"   anchor_orient value: {anchor_orient}")
    print(f"   joint_pos_hist shape: {joint_pos_hist.shape}")

    # 4. Encoder推理
    print(f"\\n4. Encoder推理:")
    try:
        enc_inputs = {
            self._encoder.get_inputs()[0].name: smpl_joints_in,
            self._encoder.get_inputs()[1].name: anchor_orient,
            self._encoder.get_inputs()[2].name: joint_pos_hist,
        }
        print(f"   输入名称: {list(enc_inputs.keys())}")

        latent = self._encoder.run(None, enc_inputs)[0]
        print(f"   ✓ Encoder推理成功")
        print(f"   latent shape: {latent.shape}")
        print(f"   latent 前10个值: {latent.flatten()[:10]}")
        self._latent = latent
    except Exception as e:
        print(f"   ✗ Encoder推理失败: {e}")
        import traceback
        traceback.print_exc()
        return self._sonic_default_np.copy()

    # 5. 准备Decoder输入
    print(f"\\n5. Decoder输入:")
    robot = self.env.scene["robot"].data
    ang_vel   = robot.root_ang_vel_b[0].cpu().numpy()
    proj_grav = robot.projected_gravity_b[0].cpu().numpy()
    joint_pos_sonic = robot.joint_pos[0, self._sonic_idx].cpu().numpy()
    joint_vel_sonic = robot.joint_vel[0, self._sonic_idx].cpu().numpy()
    dof_delta = joint_pos_sonic - self._sonic_default_np

    proprio = np.concatenate([
        ang_vel * 0.25,
        proj_grav,
        dof_delta,
        joint_vel_sonic * 0.05,
    ]).astype(np.float32)[np.newaxis]

    print(f"   ang_vel: {ang_vel}")
    print(f"   proj_grav: {proj_grav}")
    print(f"   dof_delta 前5个: {dof_delta[:5]}")
    print(f"   proprio shape: {proprio.shape}")

    # 6. Decoder推理
    print(f"\\n6. Decoder推理:")
    try:
        dec_inputs = {
            self._decoder.get_inputs()[0].name: latent,
            self._decoder.get_inputs()[1].name: proprio,
        }
        print(f"   输入名称: {list(dec_inputs.keys())}")

        action_sonic = self._decoder.run(None, dec_inputs)[0]
        print(f"   ✓ Decoder推理成功")
        print(f"   action_sonic shape: {action_sonic.shape}")

        raw_sonic = action_sonic.flatten()[:29]
        print(f"   raw_sonic 前5个: {raw_sonic[:5]}")

        # 后处理
        target_sonic = raw_sonic * 0.25 + self._sonic_default_np
        print(f"   target_sonic 前5个: {target_sonic[:5]}")

        print(f"\\n7. 结果:")
        print(f"   ✓ 推理成功")
        print(f"   target_sonic: {target_sonic}")

        return target_sonic.astype(np.float32)

    except Exception as e:
        print(f"   ✗ Decoder推理失败: {e}")
        import traceback
        traceback.print_exc()
        return self._sonic_default_np.copy()
'''

    return patch_code

def main():
    print("="*70)
    print("SONIC 推理诊断补丁生成器")
    print("="*70)

    patch = create_debug_patch()

    # 保存到文件
    with open('sonic_debug_patch.py', 'w') as f:
        f.write(patch)

    print("\\n✓ 调试补丁已生成: sonic_debug_patch.py")
    print("\\n使用方法:")
    print("1. 将 sonic_debug_patch.py 中的代码复制")
    print("2. 在 action_provider_sonic.py 中:")
    print("   - 找到 _run_gear_sonic() 函数")
    print("   - 将函数内容替换为补丁代码")
    print("3. 重新运行 run_sonic.sh")
    print("4. 查看详细的调试输出")

    print("\\n关键检查点:")
    print("  1. 模型是否正确加载")
    print("  2. SMPL数据是否有效（_smpl_data_valid）")
    print("  3. Encoder输入的形状是否正确")
    print("  4. anchor_orient 是否是 (1, 4) 而不是 (1, 24, 4)")
    print("  5. Encoder/Decoder推理是否成功")

    print("\\n" + "="*70)

if __name__ == "__main__":
    main()