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

# Layout: 與 move_football_g1_29dof_dex3_hw_env_cfg 中的 robot init_pos 對齊
# 修改以下參數可調整實驗佈局（單位：米）
ROBOT_INIT_X = -3.9
ROBOT_INIT_Y = -2.81811
GOAL_DISTANCE = 6.0  # 球門與機器人的距離（機器人朝 +X，改此值即可調整射門距離）
BALL_DISTANCE = 1.0  # 足球在機器人前方距離（射門時腳前擺放）
GOAL_Z = 0.7  # 球門高度（Z 座標）：若球門懸空則調低，直至貼地為止
GOAL_CENTER_Y_OFFSET = 2.5  # 若模型原點在邊角，調整此值使機器人對準球門中心（負值=向左移球門，正值=向右移）

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
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
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

    # Football - 在機器人前方 BALL_DISTANCE 處
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + BALL_DISTANCE, ROBOT_INIT_Y, 0.11],
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

    # Goal net - 機器人正前方 GOAL_DISTANCE，Y 軸用 GOAL_CENTER_Y_OFFSET 微調對準球門中心
    goal_net = AssetBaseCfg(
        prim_path="/World/envs/env_.*/GoalNet",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + GOAL_DISTANCE, ROBOT_INIT_Y + GOAL_CENTER_Y_OFFSET, GOAL_Z],
            rot=[-0.7071, 0.0, 0.0, 0.7071],  # 270° around Z: goal opening faces robot
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
