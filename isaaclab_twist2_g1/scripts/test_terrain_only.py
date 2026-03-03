#!/usr/bin/env python3
"""
Minimal terrain generation test - no robot required.
Tests terrain generation with IsaacLab terrain system in headless mode.

Usage:
    cd isaaclab_twist2_g1
    python scripts/test_terrain_only.py --headless
    python scripts/test_terrain_only.py --headless --terrain_type wave
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
parser = argparse.ArgumentParser(description="Test terrain generation only")
parser.add_argument("--num_steps", type=int, default=50, help="Number of simulation steps")
parser.add_argument("--terrain_type", type=str, default="flat", 
                    choices=["flat", "slope", "stairs", "pyramid", "wave", "stepping_stones", "gap"],
                    help="Type of terrain to test")

# Add IsaacLab launcher args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Now import IsaacLab modules
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# Import terrain configs - direct import to avoid DDS dependencies
sys.path.insert(0, os.path.join(project_root, "tasks"))
from common_terrains import HumanoidTerrainGeneratorCfg
from common_terrains.height_field import (
    FlatTerrainCfg,
    SlopeTerrainCfg,
    StairsTerrainCfg,
    PyramidStairsTerrainCfg,
    WaveTerrainCfg,
    SteppingStonesTerrainCfg,
    GapTerrainCfg,
)


def get_terrain_cfg(terrain_type: str):
    """Get terrain configuration based on type."""
    terrain_configs = {
        "flat": FlatTerrainCfg(
            size=(8.0, 8.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            noise_scale=0.02,
            noise_frequency=10,
        ),
        "slope": SlopeTerrainCfg(
            size=(8.0, 8.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_range=(0.1, 0.25),
            platform_width=2.0,
        ),
        "stairs": StairsTerrainCfg(
            size=(8.0, 8.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            step_height_range=(0.1, 0.15),
            step_width=0.3,
            platform_width=2.0,
        ),
        "pyramid": PyramidStairsTerrainCfg(
            size=(8.0, 8.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            step_height_range=(0.05, 0.1),
            step_width=0.4,
            platform_width=1.5,
        ),
        "wave": WaveTerrainCfg(
            size=(8.0, 8.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            amplitude_range=(0.02, 0.08),
            num_waves=4,
        ),
        "stepping_stones": SteppingStonesTerrainCfg(
            size=(8.0, 8.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            stone_height_range=(0.05, 0.1),
            stone_size_range=(0.4, 0.6),
            stone_distance_range=(0.1, 0.2),
            platform_width=2.0,
            depth=-0.3,
        ),
        "gap": GapTerrainCfg(
            size=(8.0, 8.0),
            horizontal_scale=0.1,
            vertical_scale=0.005,
            gap_width_range=(0.3, 0.5),
            gap_depth=0.4,
            platform_width=2.0,
        ),
    }
    return terrain_configs.get(terrain_type, terrain_configs["flat"])


@configclass
class TerrainOnlySceneCfg(InteractiveSceneCfg):
    """Minimal scene with only terrain and lighting."""
    
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HumanoidTerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=0.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "terrain": FlatTerrainCfg(
                    size=(8.0, 8.0),
                    horizontal_scale=0.1,
                    vertical_scale=0.005,
                ),
            },
            curriculum=False,
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0
        ),
    )


def main():
    """Test terrain generation only."""
    
    print("="*60)
    print(f"Terrain Generation Test: {args.terrain_type}")
    print("="*60)
    
    # Get terrain configuration
    terrain_cfg = get_terrain_cfg(args.terrain_type)
    print(f"\n[1] Terrain configuration:")
    print(f"   Type: {args.terrain_type}")
    print(f"   Size: {terrain_cfg.size}")
    print(f"   H-scale: {terrain_cfg.horizontal_scale}")
    print(f"   V-scale: {terrain_cfg.vertical_scale}")
    
    # Initialize physics first (required before creating scene)
    print("\n[2] Initializing physics...")
    try:
        sim_cfg = sim_utils.SimulationCfg(dt=0.01)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
        print("   ✓ Physics initialized")
    except Exception as e:
        print(f"   ✗ Failed to initialize physics: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create scene configuration
    print("\n[3] Creating scene configuration...")
    try:
        scene_cfg = TerrainOnlySceneCfg(
            num_envs=1,
            env_spacing=10.0,
            replicate_physics=True
        )
        
        # Update terrain generator with selected terrain
        scene_cfg.terrain.terrain_generator.sub_terrains = {
            "terrain": terrain_cfg,
        }
        
        print(f"   ✓ Configuration created")
    except Exception as e:
        print(f"   ✗ Failed to create configuration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create scene
    print("\n[4] Creating scene...")
    try:
        scene = InteractiveScene(scene_cfg)
        print(f"   ✓ Scene created")
        print(f"   Scene elements: {list(scene.keys())}")
    except Exception as e:
        print(f"   ✗ Failed to create scene: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Reset and run simulation
    print("\n[5] Running simulation (terrain physics)...")
    try:
        sim.reset()
        scene.reset()
        
        for step in range(args.num_steps):
            sim.step()
            scene.update(sim.get_physics_dt())
            
            if step % 10 == 0:
                print(f"   Step {step:4d}: Simulation running...")
        
        print(f"   ✓ Completed {args.num_steps} steps")
    except Exception as e:
        print(f"   ✗ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*60)
    print(f"Terrain Generation Test ({args.terrain_type}): PASSED")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        simulation_app.close()
    
    sys.exit(exit_code)
