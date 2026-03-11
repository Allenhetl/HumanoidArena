#!/usr/bin/env python3
# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Convert boxing bag OBJ to USD with physics for use in Isaac Lab.
Uses MeshConverter to add RigidBody, Collision, Mass.

Usage:
    cd isaaclab_twist2_g1
    conda activate unitree_sim_env
    python scripts/convert_boxing_bag_assets.py --headless --device cuda
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


def convert_obj_to_usd(
    obj_path: str,
    usd_path: str,
    mass=None,
    collision_approx: str = "convexDecomposition",
    make_instanceable: bool = True,
    translation=(0.0, 0.0, 0.0),
    rotation=(1.0, 0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
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
        translation=translation,
        rotation=rotation,
        scale=scale,
    )
    converter = MeshConverter(cfg)
    print(f"  -> {converter.usd_path}")
    return converter.usd_path


def main():
    print("Converting boxing bag asset (OBJ -> USD with physics)...")

    # Boxing bag: typical heavy bag mass 25-40 kg (we use 35 kg for stable swing)
    # OBJ 平躺時長軸多為 X：繞 Y 軸 90° 使其直立，quat (w,x,y,z)
    bag_rotation = (1, 0.0, 0.0, 0.0)  # 90° around Y: lay(X) -> stand(Z)
    bag_obj = os.path.join(project_root, "assets/boxing_bag/frfrstnpnchbg.obj")
    bag_usd = os.path.join(project_root, "assets/boxing_bag/boxing_bag_physics.usd")
    print("1. Boxing bag (rotation baked: 90° Y for upright):")
    convert_obj_to_usd(bag_obj, bag_usd, mass=35.0, rotation=bag_rotation)

    print("Done. USD file:")
    print(f"  - {bag_usd}")


if __name__ == "__main__":
    main()
    simulation_app.close()
