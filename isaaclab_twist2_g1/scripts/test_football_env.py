#!/usr/bin/env python3
# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Football environment test script.
Tests that the G1 29DOF Dex3 wholebody + football scene can be created and run in headless mode.

Usage:
    cd isaaclab_twist2_g1
    conda activate HumanoidArena-xzk
    python scripts/test_football_env.py --headless --device cuda

    # With rendering (requires display)
    python scripts/test_football_env.py --device cuda
"""

import argparse
import os
import sys

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

# Parse arguments
parser = argparse.ArgumentParser(description="Test football environment (G1 29DOF Dex3 + football)")
parser.add_argument("--num_steps", type=int, default=100, help="Number of simulation steps")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

# Add IsaacLab launcher args (includes --device, --headless, etc.)
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import IsaacLab modules
import gymnasium as gym
import torch

# Import task to register gym env
import tasks.g1_tasks.move_football_g1_29dof_dex3_wholebody  # noqa: F401

# Import parse_cfg for loading env config (required by IsaacLab gym envs)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    """Test football environment."""

    print("=" * 60)
    print("Football Environment Test (Isaac-Move-Football-G129-Dex3-Wholebody)")
    print("=" * 60)

    # Create environment
    print("\n[1] Creating environment...")
    task_name = "Isaac-Move-Football-G129-Dex3-Wholebody"
    try:
        env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=args.num_envs)
        env = gym.make(task_name, cfg=env_cfg)
        print("   ✓ Environment created")
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Check scene contents
    print("\n[2] Checking scene contents...")
    try:
        scene_keys = list(env.unwrapped.scene.keys())
        print(f"   Scene contains {len(scene_keys)} elements:")
        for key in scene_keys:
            print(f"      - {key}")

        if "object" in scene_keys:
            obj = env.unwrapped.scene["object"]
            pos = obj.data.root_pos_w[0].cpu().numpy()
            print(f"\n   Football (object) initial position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        if "goal_net" in scene_keys:
            print(f"   Goal net (goal_net) loaded")
        if "robot" in scene_keys:
            robot = env.unwrapped.scene["robot"]
            pos = robot.data.root_pos_w[0].cpu().numpy()
            print(f"   Robot initial position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        print("   ✓ Scene contents verified")
    except Exception as e:
        print(f"   ✗ Failed to check scene: {e}")
        import traceback

        traceback.print_exc()

    # Reset and run simulation
    print("\n[3] Running simulation...")
    try:
        obs, _ = env.reset()
        print(f"   ✓ Environment reset")
        if isinstance(obs, dict):
            print(f"   Observation keys: {list(obs.keys())}")
        else:
            print(f"   Observation type: {type(obs).__name__}")

        action_space = env.unwrapped.action_space
        action_dim = action_space.shape[-1]  # last dim for batched (num_envs, dim)
        num_envs = env.unwrapped.num_envs
        device = env.unwrapped.device

        for step in range(args.num_steps):
            # Sample random action (small gaussian for stability)
            # action must be (num_envs, action_dim) for ManagerBasedRLEnv
            action = torch.randn(num_envs, action_dim, device=device) * 0.01
            obs, reward, terminated, truncated, info = env.step(action)

            r = reward.item() if hasattr(reward, "item") else float(reward)
            done = bool((terminated | truncated).any().item()) if hasattr(terminated, "any") else bool(terminated or truncated)

            if step % 20 == 0:
                print(f"   Step {step:4d}: reward={r:.3f}, done={done}")

        print(f"   ✓ Completed {args.num_steps} simulation steps")
    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Cleanup
    print("\n[4] Cleaning up...")
    try:
        env.close()
        print("   ✓ Environment closed")
    except Exception as e:
        print(f"   ✗ Cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("Football Environment Test: PASSED")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()

    sys.exit(exit_code)
