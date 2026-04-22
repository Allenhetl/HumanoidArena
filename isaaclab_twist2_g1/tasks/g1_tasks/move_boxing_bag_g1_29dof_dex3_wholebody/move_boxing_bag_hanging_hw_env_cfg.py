# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Hanging boxing bag variant: fixed anchor + swinging bag."""

from .move_boxing_bag_g1_29dof_dex3_hw_env_cfg import MoveBoxingBagG129Dex3WholebodyEnvCfg
from tasks.common_scene.base_scene_hanging_boxing_bag_cfg_wholebody import (
    TableHangingBoxingBagSceneCfgWH,
    ROBOT_INIT_X,
    ROBOT_INIT_Y,
    ROBOT_INIT_Z,
)
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from tasks.common_config import G1RobotPresets, CameraPresets
from isaaclab.utils import configclass


@configclass
class HangingBoxingBagSceneCfg(TableHangingBoxingBagSceneCfgWH):
    """Hanging boxing bag scene: anchor fixed, bag swings when punched."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=(ROBOT_INIT_X, ROBOT_INIT_Y, ROBOT_INIT_Z),
        init_rot=(0.7071, 0.0, 0.0, 0.7071),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.g1_front_camera()
    world_camera = CameraPresets.g1_world_camera()


@configclass
class MoveBoxingBagHangingG129Dex3WholebodyEnvCfg(MoveBoxingBagG129Dex3WholebodyEnvCfg):
    """Boxing bag task with hanging bag (anchor fixed, bag swings)."""

    scene: HangingBoxingBagSceneCfg = HangingBoxingBagSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    def __post_init__(self):
        super().__post_init__()
        self.event_manager.unregister("reset_object_self")
