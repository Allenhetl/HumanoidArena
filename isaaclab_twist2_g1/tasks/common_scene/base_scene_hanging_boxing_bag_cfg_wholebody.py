# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Hanging boxing bag scene configuration for G1 wholebody tasks.
Uses Articulation (anchor + revolute joint + bag): fixed anchor, bag swings when punched.
"""
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_scene.base_scene_pickplace_cylindercfg_wholebody import TableCylinderSceneCfgWH

project_root = os.environ.get("PROJECT_ROOT")

# =============================================================================
# 掛取錨點與沙袋位置調整變數（統一在此修改）
# =============================================================================
# 機器人初始位置
ROBOT_INIT_X = -1.9
ROBOT_INIT_Y = -5.2
ROBOT_INIT_Z = 0.8

# 沙袋相對於機器人的偏移（機器人朝 +Y，沙袋在正前方）
BAG_DISTANCE = 1.8
BAG_OFFSET_X = 0.0
BAG_OFFSET_Y = BAG_DISTANCE

# 錨點高度 [m]：天花板掛點 Z 座標，倒吊沙袋從此垂下
HANGING_ANCHOR_Z = 1.6

# 錨點世界座標 = (ROBOT_INIT_X + BAG_OFFSET_X, ROBOT_INIT_Y + BAG_OFFSET_Y, HANGING_ANCHOR_Z)
#
# 錨點到沙袋掛載點的距離：修改 hanging_bag.urdf 中 joint 的
#   <origin xyz="0 0 -0.02" ... /> 的 Z 分量（負值表示向下，如 -0.05 = 5cm）


@configclass
class TableHangingBoxingBagSceneCfgWH(TableCylinderSceneCfgWH):
    """Hanging boxing bag scene: fixed anchor + revolute joint + bag."""

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

    box = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-50.0, -50.0, -10.0),
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

    # Hanging bag: Articulation (anchor fixed, bag swings via revolute joint)
    object = ArticulationCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/hanging_boxing_bag/hanging_bag_articulation.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=[ROBOT_INIT_X + BAG_OFFSET_X, ROBOT_INIT_Y + BAG_OFFSET_Y, HANGING_ANCHOR_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
            joint_pos={"swing_joint": 0.0},
        ),
        actuators={
            "swing": ImplicitActuatorCfg(
                joint_names_expr=["swing_joint"],
                effort_limit_sim=0.0,
                velocity_limit_sim=0.0,
                stiffness=0.0,
                damping=0.3,
            ),
        },
    )
