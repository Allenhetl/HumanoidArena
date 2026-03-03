# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0
"""
Visual zone navigation task environment configuration.
Robot navigates to target zones while avoiding forbidden zones using vision.
"""

import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg

from . import mdp

from tasks.common_config import G1RobotPresets, CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# Import visual zones scene configuration
from tasks.common_scene.base_scene_visual_zones import VisualZonesSceneCfg


##
# Scene definition
##
@configclass
class VisualZoneSceneCfg(VisualZonesSceneCfg):
    """Visual zone navigation scene configuration.
    
    Inherits from VisualZonesSceneCfg which provides:
    - Warehouse environment
    - Ground plane
    - Target zones (green)
    - Forbidden zones (red)
    - Lighting
    
    This class adds:
    - G1 robot with Dex3 hands
    - Contact sensors
    - Cameras for vision-based navigation
    """

    # G1 robot with Dex3 dexterous hands - positioned at origin
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=(0.0, 0.0, 0.8),  # Start position (before target zone)
        init_rot=(1.0, 0.0, 0.0, 0.0)  # Facing forward (+Y direction)
    )

    # Contact sensors for detecting collisions
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False
    )

    # Camera configurations for vision-based navigation
    front_camera = CameraPresets.g1_front_camera()
    world_camera = CameraPresets.g1_world_camera()


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action configuration for robot control.
    
    Uses direct joint position control for the full body.
    """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation configuration for visual zone navigation.
    
    Includes:
    - Robot body joint states
    - Robot gripper (Dex3) joint states
    - Camera images for vision-based navigation
    - Robot base pose for reward computation
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observation group."""

        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gripper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)
        robot_base_pose = ObsTerm(func=mdp.get_robot_base_pose)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination conditions for visual zone navigation."""
    
    # Episode timeout
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    
    # Terminate if robot falls (base height too low)
    robot_fall = DoneTerm(func=mdp.check_robot_fall)


@configclass
class RewardsCfg:
    """Reward configuration for visual zone navigation.
    
    Rewards:
    - Positive reward for reaching target zone
    - Negative reward for entering forbidden zone
    - Small reward for moving towards target
    - Penalty for falling
    """
    
    # Main navigation rewards
    reach_target = RewTerm(func=mdp.reward_reach_target_zone, weight=10.0)
    avoid_forbidden = RewTerm(func=mdp.reward_avoid_forbidden_zone, weight=-5.0)
    
    # Progress rewards
    move_towards_target = RewTerm(func=mdp.reward_move_towards_target, weight=1.0)
    
    # Safety penalties
    fall_penalty = RewTerm(func=mdp.penalty_robot_fall, weight=-10.0)
    
    # Energy efficiency
    action_penalty = RewTerm(func=mdp.penalty_action_magnitude, weight=-0.01)


@configclass
class EventCfg:
    """Event configuration for environment resets."""
    pass


@configclass
class VisualZoneG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """Visual zone navigation environment configuration.
    
    Task: Navigate to target zones (green) while avoiding forbidden zones (red)
    using visual observations from the robot's camera.
    """

    # Scene settings
    scene: VisualZoneSceneCfg = VisualZoneSceneCfg(
        num_envs=1,
        env_spacing=5.0,
        replicate_physics=True
    )

    # MDP settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization settings."""
        # Timing settings
        self.decimation = 10  # Policy runs at 100Hz (1000Hz / 10)
        self.episode_length_s = 30.0  # 30 seconds per episode
        
        # Simulation settings
        self.sim.dt = 0.001  # 1ms timestep (1000Hz physics)
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        
        # Physics settings
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Physics material properties
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        # Event manager for custom events
        self.event_manager = SimpleEventManager()

        # Register reset events
        self.event_manager.register("reset_robot_pose", SimpleEvent(
            func=lambda env: base_mdp.reset_root_state_uniform(
                env,
                torch.arange(env.num_envs, device=env.device),
                pose_range={"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
                velocity_range={},
                asset_cfg=SceneEntityCfg("robot"),
            )
        ))

        self.event_manager.register("reset_scene", SimpleEvent(
            func=lambda env: base_mdp.reset_scene_to_default(
                env,
                torch.arange(env.num_envs, device=env.device)
            )
        ))
