# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Football (soccer ball) scene configuration for G1 wholebody tasks.
Uses SphereCfg with FIFA standard football dimensions and physical properties.
Uses ISAAC_NUCLEUS Simple_Warehouse for room (no local project assets required).
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tasks.common_scene.base_scene_pickplace_cylindercfg_wholebody import TableCylinderSceneCfgWH


# FIFA standard football specifications:
# - Circumference: 68-70 cm -> diameter ~22 cm, radius = 0.11 m
# - Mass: 410-450 g -> 0.43 kg
# - Restitution (bounciness): typical soccer ball ~0.7-0.8


@configclass
class TableFootballSceneCfgWH(TableCylinderSceneCfgWH):
    """Football table scene configuration.
    Extends TableCylinderSceneCfgWH and adds a FIFA-standard football as the manipulable object.
    Uses ISAAC_NUCLEUS Simple_Warehouse for room (no local project assets required).
    """

    # Override room to use Isaac Nucleus warehouse (avoids dependency on local small_warehouse USD)
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        ),
    )

    # Override box to use simple CuboidCfg (avoids dependency on local CubeBox USD)
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-3.3, -3.06, 0.9],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
    )

    # Football object - FIFA standard dimensions and physics
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-3.3, -3.06, 0.95],  # Initial position (slightly above ground for visibility)
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.SphereCfg(
            radius=0.11,  # FIFA standard: diameter 22 cm -> radius 0.11 m
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),  # FIFA: 410-450 g
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.9, 0.85),  # Light football color
                metallic=0.0,
                roughness=0.6,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="max",
                static_friction=0.6,  # Typical ball-on-ground friction
                dynamic_friction=0.5,
                restitution=0.75,  # Soccer ball bounce
            ),
        ),
    )
