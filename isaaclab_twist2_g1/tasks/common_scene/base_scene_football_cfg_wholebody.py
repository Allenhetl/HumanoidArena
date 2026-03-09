# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Football (soccer ball) scene configuration for G1 wholebody tasks.
Uses textured soccer ball USD (ImageToStl.com_Soccer+Ball.usdc) and goal net USD.
Uses ISAAC_NUCLEUS Simple_Warehouse for room (no local project assets required).
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from tasks.common_scene.base_scene_pickplace_cylindercfg_wholebody import TableCylinderSceneCfgWH

project_root = os.environ.get("PROJECT_ROOT")

# Layout: 統一在此調整機器人、球、球門的初始位置
# 機器人已向左旋轉 90° (init_rot=(0.7071,0,0,0.7071))，朝 +Y 方向
# 球與球門的相對位置已同步旋轉，保持與機器人的相對位姿
ROBOT_INIT_X = -1.9
ROBOT_INIT_Y = -5.2
ROBOT_INIT_Z = 0.8  # 機器人站立高度
GOAL_DISTANCE = 6.0  # 球門與機器人距離（沿機器人朝向）
BALL_DISTANCE = 1.0  # 足球在機器人前方距離
GOAL_Z = 0.7  # 球門高度（Z 座標）
GOAL_CENTER_Y_OFFSET = 2.5  # 球門中心相對於機器人朝向的橫向偏移

# 90° 左旋後：原 (dx,dy) → (-dy, dx)，機器人朝 +Y
# Ball: 原 (1,0) → (0, 1)
# Goal: 原 (6, 2.5) → (-2.5, 6)
BALL_OFFSET_X = 0.0  # -dy
BALL_OFFSET_Y = BALL_DISTANCE  # dx
GOAL_OFFSET_X = -GOAL_CENTER_Y_OFFSET  # -dy
GOAL_OFFSET_Y = GOAL_DISTANCE  # dx

# FIFA standard football specifications:
# - Circumference: 68-70 cm -> diameter ~22 cm, radius = 0.11 m
# - Mass: 410-450 g -> 0.43 kg
# - Restitution (bounciness): typical soccer ball ~0.7-0.8
# ImageToStl STL/USD typically uses mm units -> scale 0.001 for 220mm ball


@configclass
class TableFootballSceneCfgWH(TableCylinderSceneCfgWH):
    """Football table scene configuration.
    Extends TableCylinderSceneCfgWH with textured soccer ball USD and goal net.
    Uses ISAAC_NUCLEUS Simple_Warehouse for room.
    """

    # Override room to use Isaac Nucleus warehouse (avoids dependency on local small_warehouse USD)
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin_boxtarget.usd",
        ),
    )

    # Hide box (no table needed for football task - ball on floor)
    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-50.0, -50.0, -10.0),  # Move out of view
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.2)),
        ),
    )

    # Football - 在機器人前方 BALL_DISTANCE 處（隨機器人 90° 左旋同步旋轉）
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + BALL_OFFSET_X, ROBOT_INIT_Y + BALL_OFFSET_Y, 0.11],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/football/soccer_ball_physics.usd",
            scale=(1.0, 1.0, 1.0),  # OBJ in meters -> FIFA ball ~0.22m
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.43),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    # Goal net - 機器人正前方 GOAL_DISTANCE，隨機器人 90° 左旋同步旋轉，球門開口朝向機器人
    goal_net = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalNet",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + GOAL_OFFSET_X, ROBOT_INIT_Y + GOAL_OFFSET_Y, GOAL_Z],
            rot=[1.0, 0.0, 0.0, 0.0],  # 球網在原有基礎上再向左旋轉 90°
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/football_net/football_goal_physics.usd",
            scale=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
        ),
    )
