# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Visual zones scene configuration.
Provides scene with visual markers for vision-based navigation tasks.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tasks.common_config import CameraBaseCfg
from tasks.common_terrains.visual_markers import (
    RectangleZoneCfg,
    CircleZoneCfg,
    create_zone_boundary_assets,
)

import os
project_root = os.environ.get("PROJECT_ROOT")


@configclass
class VisualZonesSceneCfg(InteractiveSceneCfg):
    """Scene configuration with visual zone markers for navigation tasks.
    
    This scene includes:
    - Warehouse environment
    - Ground plane
    - Multiple visual zones (target, forbidden, path markers)
    - Lighting
    """
    
    # Room/warehouse environment (Isaac Nucleus - no local assets required)
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
        ),
    )
    
    # Ground plane (backup if warehouse doesn't have one)
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        ),
    )
    
    # Lighting
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0
        ),
    )
    
    # Target zone (green) - robot should reach this area
    # Using flat prim paths to avoid nested prim creation issues
    target_zone_line_front = AssetBaseCfg(
        prim_path="/World/envs/env_.*/target_line_front",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 2.5, 0.005],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.02, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    target_zone_line_back = AssetBaseCfg(
        prim_path="/World/envs/env_.*/target_line_back",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 3.5, 0.005],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.02, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    target_zone_line_left = AssetBaseCfg(
        prim_path="/World/envs/env_.*/target_line_left",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.5, 3.0, 0.005],
            rot=[0.7071, 0.0, 0.0, 0.7071],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.02, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    target_zone_line_right = AssetBaseCfg(
        prim_path="/World/envs/env_.*/target_line_right",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 3.0, 0.005],
            rot=[0.7071, 0.0, 0.0, 0.7071],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 0.02, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    # Target zone corner markers (yellow cones)
    target_corner_1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/target_corner_1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.5, 2.5, 0.075],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.ConeCfg(
            radius=0.05,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    target_corner_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/target_corner_2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 2.5, 0.075],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.ConeCfg(
            radius=0.05,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    target_corner_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/target_corner_3",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 3.5, 0.075],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.ConeCfg(
            radius=0.05,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    target_corner_4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/target_corner_4",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.5, 3.5, 0.075],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.ConeCfg(
            radius=0.05,
            height=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    # Forbidden zone (red) - robot should avoid this area
    forbidden_zone_line_front = AssetBaseCfg(
        prim_path="/World/envs/env_.*/forbidden_line_front",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1.5, 2.5, 0.005],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.03, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    forbidden_zone_line_back = AssetBaseCfg(
        prim_path="/World/envs/env_.*/forbidden_line_back",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1.5, 3.3, 0.005],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.03, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    forbidden_zone_line_left = AssetBaseCfg(
        prim_path="/World/envs/env_.*/forbidden_line_left",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1.1, 2.9, 0.005],
            rot=[0.7071, 0.0, 0.0, 0.7071],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.03, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    forbidden_zone_line_right = AssetBaseCfg(
        prim_path="/World/envs/env_.*/forbidden_line_right",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1.9, 2.9, 0.005],
            rot=[0.7071, 0.0, 0.0, 0.7071],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.03, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0),
                metallic=0.0,
            ),
        ),
    )
    
    # Forbidden zone corner markers (red cylinders)
    forbidden_corner_1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/forbidden_corner_1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1.1, 2.5, 0.1],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
                metallic=0.0,
            ),
        ),
    )
    
    forbidden_corner_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/forbidden_corner_2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1.9, 2.5, 0.1],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
                metallic=0.0,
            ),
        ),
    )
    
    forbidden_corner_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/forbidden_corner_3",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1.9, 3.3, 0.1],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
                metallic=0.0,
            ),
        ),
    )
    
    forbidden_corner_4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/forbidden_corner_4",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[1.1, 3.3, 0.1],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.04,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
                metallic=0.0,
            ),
        ),
    )
    
    # World camera for third-person view
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(0.0, -2.0, 2.5),
        rot_offset=(-0.3, 0.0, 0.0, 0.95),
        focal_length=18.0
    )
