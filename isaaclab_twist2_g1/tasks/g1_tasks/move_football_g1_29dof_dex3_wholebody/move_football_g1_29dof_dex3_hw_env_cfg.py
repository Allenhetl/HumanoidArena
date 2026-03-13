# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg

from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_football_cfg_wholebody import (
    TableFootballSceneCfgWH,
    ROBOT_INIT_X,
    ROBOT_INIT_Y,
    ROBOT_INIT_Z,
)


##
# Scene definition
##


@configclass
class FootballTableSceneCfg(TableFootballSceneCfgWH):
    """Football table scene with G1 29DOF Dex3 wholebody robot."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=(ROBOT_INIT_X, ROBOT_INIT_Y, ROBOT_INIT_Z),
        init_rot=(0.7071, 0.0, 0.0, 0.7071),  # 向左旋轉 90° (繞 Z 軸)
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.g1_front_camera()
    # world_camera = CameraPresets.g1_world_camera()


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Joint position action configuration."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""

        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    pass


@configclass
class MoveFootballG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """
    Environment configuration for G1 29DOF Dex3 wholebody robot with football task.
    """

    scene: FootballTableSceneCfg = FootballTableSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 10  # Policy frequency: 1000Hz / 10 = 100Hz (matches TWIST2 training)
        self.episode_length_s = 20.0

        self.sim.dt = 0.002 # 1ms timestep = 1000Hz physics (matches cylinder task)
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 1
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        # 全局 PhysX 求解器设置
        # self.sim.physx.solver_type = 1  # TGS solver
        # self.sim.physx.min_position_iteration_count = 16
        # self.sim.physx.max_position_iteration_count = 16  # 增加
        # self.sim.physx.min_velocity_iteration_count = 8
        # self.sim.physx.max_velocity_iteration_count = 8  # 增加

        self.event_manager = SimpleEventManager()

        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_root_state_uniform(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                    pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
                    velocity_range={},
                    asset_cfg=SceneEntityCfg("object"),
                )
            ),
        )

        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_scene_to_default(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                )
            ),
        )
