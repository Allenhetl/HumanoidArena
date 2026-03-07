#!/usr/bin/env python3
# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Convert football and goal OBJ to USD with physics for use in Isaac Lab.
Uses MeshConverter to add RigidBody, Collision, Mass.

Usage:
    cd isaaclab_twist2_g1
    conda activate unitree_sim_env
    python scripts/convert_football_assets.py --headless --device cuda
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.environ["PROJECT_ROOT"] = project_root

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path


def convert_obj_to_usd(
    obj_path: str,
    usd_path: str,
    mass=None,
    collision_approx: str = "convexDecomposition",
    make_instanceable: bool = True,
) -> str:
    obj_path = os.path.abspath(obj_path)
    usd_path = os.path.abspath(usd_path)
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ not found: {obj_path}")
    os.makedirs(os.path.dirname(usd_path), exist_ok=True)

    mass_props = schemas_cfg.MassPropertiesCfg(mass=mass) if mass is not None else None
    rigid_props = schemas_cfg.RigidBodyPropertiesCfg() if mass is not None else None
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=collision_approx != "none"
    )

    cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=obj_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(usd_path),
        usd_file_name=os.path.basename(usd_path),
        make_instanceable=make_instanceable,
        collision_approximation=collision_approx,
    )
    converter = MeshConverter(cfg)
    print(f"  -> {converter.usd_path}")
    return converter.usd_path


def main():
    print("Converting football assets (OBJ -> USD with physics)...")

    # Soccer ball: FIFA mass 0.43 kg
    ball_obj = os.path.join(project_root, "assets/football/standard soccer ball/Soccer Ball.obj")
    ball_usd = os.path.join(project_root, "assets/football/soccer_ball_physics.usd")
    print("1. Soccer ball:")
    convert_obj_to_usd(ball_obj, ball_usd, mass=0.43)

    # Goal: kinematic, need rigid body for collision (mass=1 unused when kinematic)
    goal_obj = os.path.join(project_root, "assets/football_net/Football Goal/football goal.obj")
    goal_usd = os.path.join(project_root, "assets/football_net/football_goal_physics.usd")
    print("2. Football goal:")
    convert_obj_to_usd(goal_obj, goal_usd, mass=1.0)

    print("Done. USD files:")
    print(f"  - {ball_usd}")
    print(f"  - {goal_usd}")


if __name__ == "__main__":
    main()
    simulation_app.close()
