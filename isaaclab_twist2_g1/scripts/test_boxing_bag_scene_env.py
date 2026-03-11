#!/usr/bin/env python3
# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
拳擊沙袋場景測試腳本（含 headless 模式）

用法:
    cd isaaclab_twist2_g1

    # Headless 測試（無視窗，需加 --enable_cameras 啟用相機）
    python scripts/test_boxing_bag_scene_env.py --headless --enable_cameras --device cuda --num_steps 100

    # 帶渲染可視化
    python scripts/test_boxing_bag_scene_env.py --enable_cameras --device cuda

    # 自訂步數
    python scripts/test_boxing_bag_scene_env.py --headless --enable_cameras --device cuda --num_steps 200
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

parser = argparse.ArgumentParser(description="Headless test for boxing bag scene")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps (ignored if --no_limit)")
parser.add_argument("--no_limit", action="store_true", help="Run until Ctrl+C")
parser.add_argument("--num_envs", type=int, default=1)

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from tasks.g1_tasks.move_boxing_bag_g1_29dof_dex3_wholebody.move_boxing_bag_g1_29dof_dex3_hw_env_cfg import (
    MoveBoxingBagG129Dex3WholebodyEnvCfg,
)


def main():
    print("=" * 60)
    print("Boxing Bag Scene Headless Test")
    print("=" * 60)

    print("\n[1] Creating environment configuration...")
    try:
        env_cfg = MoveBoxingBagG129Dex3WholebodyEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        print("   ✓ Configuration created")
        print(f"   - Num envs: {env_cfg.scene.num_envs}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n[2] Creating environment...")
    try:
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print("   ✓ Environment created")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n[3] Scene contents:", list(env.scene.keys()))

    print("\n[4] Running simulation...")
    try:
        env.reset()
        print("   ✓ Environment reset")
        if args.no_limit:
            print("   Running until Ctrl+C...")
        step = 0
        try:
            while True:
                env.sim.step(render=True)
                if args.no_limit and step % 100 == 0:
                    print(f"   Step {step}...")
                elif not args.no_limit and step % 20 == 0:
                    print(f"   Step {step:4d}: boxing bag scene running...")
                step += 1
                if not args.no_limit and step >= args.num_steps:
                    break
        except KeyboardInterrupt:
            print("\n   Stopped by user (Ctrl+C).")
        if not args.no_limit:
            print(f"   ✓ Completed {args.num_steps} steps")
    except KeyboardInterrupt:
        print("\n   Stopped by user (Ctrl+C).")
        return 0
    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    env.close()
    print("\n" + "=" * 60)
    print("Boxing Bag Headless Test: PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    sys.exit(exit_code)
