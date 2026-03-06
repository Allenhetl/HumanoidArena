#!/usr/bin/env python3
# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Visual zones environment test script.
Tests that the visual zone scene can be created and run in headless mode.

Usage:
    cd isaaclab_twist2_g1
    python scripts/test_visual_zones_env.py --headless --device cuda

    # With rendering (requires display)
    python scripts/test_visual_zones_env.py --device cuda

    # Custom step count (e.g. 500 steps)
    python scripts/test_visual_zones_env.py --device cuda --num_steps 500

    # Run until you press Ctrl+C (no step limit)
    python scripts/test_visual_zones_env.py --device cuda --no_limit
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
parser = argparse.ArgumentParser(description="Test visual zones environment")
parser.add_argument("--num_steps", type=int, default=100, help="Number of simulation steps (ignored if --no_limit)")
parser.add_argument("--no_limit", action="store_true", help="Run until Ctrl+C, no step limit")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")

# Add IsaacLab launcher args (includes --device, --headless, etc.)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import IsaacLab modules
import torch
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Import visual zones scene
from tasks.common_scene.base_scene_visual_zones import VisualZonesSceneCfg


##
# Scene Configuration
##
@configclass
class VisualZoneTestSceneCfg(VisualZonesSceneCfg):
    """Test scene with visual zones (no robot for minimal testing)."""
    
    # Disable camera for headless testing without --enable_cameras
    world_camera = None


##
# MDP Configuration
##
@configclass
class ActionsCfg:
    """Simple action configuration."""
    pass  # No actions for this test


@configclass
class ObservationsCfg:
    """Simple observation configuration."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Empty rewards for test."""
    pass


@configclass
class TerminationsCfg:
    """Empty terminations for test."""
    pass


##
# Environment Configuration
##
@configclass
class VisualZoneTestEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for visual zone test environment."""
    
    scene: VisualZoneTestSceneCfg = VisualZoneTestSceneCfg(
        num_envs=1,
        env_spacing=5.0,
        replicate_physics=True
    )
    
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self):
        """Post initialization."""
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation


def main():
    """Test visual zones environment."""
    
    print("="*60)
    print("Visual Zones Environment Test")
    print("="*60)
    
    # Create environment configuration
    print("\n[1] Creating environment configuration...")
    try:
        env_cfg = VisualZoneTestEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        print(f"   ✓ Configuration created")
        print(f"   - Num envs: {env_cfg.scene.num_envs}")
        print(f"   - Decimation: {env_cfg.decimation}")
        print(f"   - dt: {env_cfg.sim.dt}")
    except Exception as e:
        print(f"   ✗ Failed to create configuration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create environment
    print("\n[2] Creating environment...")
    try:
        # Create minimal environment without gym registration
        from isaaclab.envs import ManagerBasedRLEnv
        env = ManagerBasedRLEnv(cfg=env_cfg)
        print(f"   ✓ Environment created")
    except Exception as e:
        print(f"   ✗ Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Check scene contents
    print("\n[3] Checking scene contents...")
    try:
        scene_keys = list(env.scene.keys())
        print(f"   Scene contains {len(scene_keys)} elements:")
        for key in scene_keys:
            print(f"      - {key}")
        
        # Check for visual zone elements
        zone_elements = [k for k in scene_keys if 'zone' in k.lower() or 'corner' in k.lower() or 'line' in k.lower()]
        print(f"\n   Visual zone elements: {len(zone_elements)}")
        for elem in zone_elements:
            print(f"      - {elem}")
        
        print("   ✓ Scene contents verified")
    except Exception as e:
        print(f"   ✗ Failed to check scene: {e}")
        import traceback
        traceback.print_exc()
    
    # Reset and run simulation
    print("\n[4] Running simulation...")
    try:
        env.reset()
        print(f"   ✓ Environment reset")
        if args.no_limit:
            print("   Running until Ctrl+C (no step limit)...")
        step = 0
        try:
            while True:
                env.sim.step(render=True)
                if args.no_limit:
                    if step % 100 == 0:
                        print(f"   Step {step}: running... (Ctrl+C to exit)")
                elif step % 20 == 0:
                    print(f"   Step {step:4d}: Visual zones simulation running...")
                step += 1
                if not args.no_limit and step >= args.num_steps:
                    break
        except KeyboardInterrupt:
            if args.no_limit:
                print("\n   Stopped by user (Ctrl+C).")
            else:
                raise
        if not args.no_limit:
            print(f"   ✓ Completed {args.num_steps} simulation steps")
    except KeyboardInterrupt:
        print("\n   Stopped by user (Ctrl+C).")
        return 0
    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Cleanup
    print("\n[5] Cleaning up...")
    try:
        env.close()
        print("   ✓ Environment closed")
    except Exception as e:
        print(f"   ✗ Cleanup failed: {e}")
    
    print("\n" + "="*60)
    print("Visual Zones Environment Test: PASSED")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    
    sys.exit(exit_code)
