# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0

"""Observation functions for visual zone navigation task."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Import common observation functions
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
from tasks.common_observations.dex3_state import get_robot_dex3_joint_states
from tasks.common_observations.camera_state import get_camera_image


def get_robot_base_pose(env: "ManagerBasedEnv") -> torch.Tensor:
    """Get the robot base position and orientation.
    
    Returns:
        Tensor of shape (num_envs, 7): [x, y, z, qw, qx, qy, qz]
    """
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w  # (num_envs, 3)
    quat = robot.data.root_quat_w  # (num_envs, 4)
    return torch.cat([pos, quat], dim=-1)


def get_robot_base_velocity(env: "ManagerBasedEnv") -> torch.Tensor:
    """Get the robot base linear and angular velocity.
    
    Returns:
        Tensor of shape (num_envs, 6): [vx, vy, vz, wx, wy, wz]
    """
    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_w  # (num_envs, 3)
    ang_vel = robot.data.root_ang_vel_w  # (num_envs, 3)
    return torch.cat([lin_vel, ang_vel], dim=-1)


def get_distance_to_target_zone(env: "ManagerBasedEnv") -> torch.Tensor:
    """Compute distance from robot to target zone center.
    
    Target zone is centered at (0.0, 3.0) based on scene configuration.
    
    Returns:
        Tensor of shape (num_envs, 1): distance to target zone
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]  # (num_envs, 2) xy position
    
    # Target zone center (from scene configuration)
    target_center = torch.tensor([0.0, 3.0], device=robot_pos.device)
    
    # Compute distance
    distance = torch.norm(robot_pos - target_center, dim=-1, keepdim=True)
    return distance


def get_target_direction(env: "ManagerBasedEnv") -> torch.Tensor:
    """Get normalized direction vector from robot to target zone.
    
    Returns:
        Tensor of shape (num_envs, 2): normalized direction [dx, dy]
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]  # (num_envs, 2)
    
    # Target zone center
    target_center = torch.tensor([0.0, 3.0], device=robot_pos.device)
    
    # Direction vector
    direction = target_center - robot_pos
    
    # Normalize
    norm = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_normalized = direction / norm
    
    return direction_normalized


__all__ = [
    "get_robot_boy_joint_states",
    "get_robot_dex3_joint_states",
    "get_camera_image",
    "get_robot_base_pose",
    "get_robot_base_velocity",
    "get_distance_to_target_zone",
    "get_target_direction",
]
