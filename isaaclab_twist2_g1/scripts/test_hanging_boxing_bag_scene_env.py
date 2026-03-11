#!/usr/bin/env python3
# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
倒吊拳擊沙袋場景測試腳本

用法:
    # 1. 先轉換資產（URDF -> USD）
    python scripts/convert_hanging_boxing_bag_assets.py --headless --device cuda

    # 2. Headless 測試（無相機，較輕量）
    python scripts/test_hanging_boxing_bag_scene_env.py --headless --device cuda --num_steps 100

    # 2b. 含相機測試（若無相機能過，加 --enable_cameras 再試）
    python scripts/test_hanging_boxing_bag_scene_env.py --headless --enable_cameras --device cuda --num_steps 100

    # 3. 帶渲染可視化
    python scripts/test_hanging_boxing_bag_scene_env.py --enable_cameras --device cuda
"""

import argparse
import os
import sys
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

parser = argparse.ArgumentParser(description="Test hanging boxing bag scene")
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--no_limit", action="store_true", help="Run until Ctrl+C")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--excite_mode",
    type=str,
    default="force",
    choices=["none", "force", "joint_vel"],
    help="Excitation mode for visualizing joint center",
)
parser.add_argument("--excite_interval", type=int, default=120, help="Steps between excitations")
parser.add_argument("--excite_duration", type=int, default=10, help="Excitation duration in steps")
parser.add_argument("--excite_force", type=float, default=250.0, help="Force pulse magnitude [N]")
parser.add_argument("--excite_joint_vel", type=float, default=2.0, help="Joint velocity pulse [rad/s]")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from tasks.g1_tasks.move_boxing_bag_g1_29dof_dex3_wholebody.move_boxing_bag_hanging_hw_env_cfg import (
    MoveBoxingBagHangingG129Dex3WholebodyEnvCfg,
)


def main():
    import traceback

    print("=" * 60)
    print("Hanging Boxing Bag Scene Test")
    print("=" * 60)

    print("\n[1] Creating environment...")
    try:
        env_cfg = MoveBoxingBagHangingG129Dex3WholebodyEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        print("   ✓ Config created")

        env = ManagerBasedRLEnv(cfg=env_cfg)
        print("   ✓ Environment created")
    except Exception as e:
        print(f"\n   ✗ FAILED: {e}")
        traceback.print_exc()
        return 1

    print("\n[2] Scene contents:", list(env.scene.keys()))

    print("\n[3] Running simulation...")
    env.reset()
    obj = env.scene["object"]
    num_envs = env.scene.num_envs
    device = env.device

    swing_joint_id, swing_joint_names = obj.find_joints("swing_joint")
    if len(swing_joint_id) == 0:
        raise RuntimeError("Cannot find swing_joint in hanging bag articulation.")
    swing_joint_id = swing_joint_id[0]

    bag_body_id, bag_body_names = obj.find_bodies("bag")
    if len(bag_body_id) == 0:
        raise RuntimeError("Cannot find bag body in hanging bag articulation.")
    bag_body_id = bag_body_id[0]

    print(f"   Excitation mode: {args.excite_mode}")
    print(f"   swing_joint: {swing_joint_names[0]} (id={swing_joint_id})")
    print(f"   bag_body: {bag_body_names[0]} (id={bag_body_id})")

    force_cmd = torch.zeros((num_envs, 1, 3), device=device)
    torque_cmd = torch.zeros((num_envs, 1, 3), device=device)
    vel_kick = torch.zeros((num_envs, 1), device=device)

    step = 0
    try:
        while True:
            active = (step % args.excite_interval) < args.excite_duration

            if args.excite_mode == "force":
                force_cmd.zero_()
                torque_cmd.zero_()
                if active:
                    # Local +Y push at bag link to excite pendulum motion.
                    force_cmd[:, 0, 1] = args.excite_force
                obj.set_external_force_and_torque(
                    forces=force_cmd,
                    torques=torque_cmd,
                    body_ids=[bag_body_id],
                )
                obj.write_data_to_sim()
            elif args.excite_mode == "joint_vel":
                vel_kick.zero_()
                if active:
                    vel_kick[:, 0] = args.excite_joint_vel
                obj.write_joint_velocity_to_sim(vel_kick, joint_ids=[swing_joint_id])

            env.sim.step(render=True)
            if step % 20 == 0:
                swing_q = obj.data.joint_pos[0, swing_joint_id].item()
                swing_qd = obj.data.joint_vel[0, swing_joint_id].item()
                print(
                    f"   Step {step:4d}: q={swing_q:+.3f} rad, qd={swing_qd:+.3f} rad/s, excite={active}"
                )
            step += 1
            if not args.no_limit and step >= args.num_steps:
                break
    except KeyboardInterrupt:
        print("\n   Stopped by user (Ctrl+C).")

    if args.excite_mode == "force":
        force_cmd.zero_()
        torque_cmd.zero_()
        obj.set_external_force_and_torque(forces=force_cmd, torques=torque_cmd, body_ids=[bag_body_id])
        obj.write_data_to_sim()

    if not args.no_limit:
        print(f"   ✓ Completed {args.num_steps} steps")

    env.close()
    print("\n" + "=" * 60)
    print("Hanging Boxing Bag Test: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
