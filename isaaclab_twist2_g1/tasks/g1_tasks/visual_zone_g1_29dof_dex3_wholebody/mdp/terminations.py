# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0

"""Termination functions for visual zone navigation task."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Constants
ROBOT_FALL_HEIGHT_THRESHOLD = 0.3  # meters
TARGET_ZONE_CENTER = torch.tensor([0.0, 3.0])
TARGET_ZONE_SIZE = torch.tensor([1.0, 1.0])


def check_robot_fall(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if robot has fallen (base height too low).
    
    Returns:
        Boolean tensor (num_envs,): True if robot has fallen
    """
    robot = env.scene["robot"]
    robot_height = robot.data.root_pos_w[:, 2]
    
    has_fallen = robot_height < ROBOT_FALL_HEIGHT_THRESHOLD
    
    return has_fallen


def check_reached_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if robot has reached the target zone.
    
    Returns:
        Boolean tensor (num_envs,): True if robot is in target zone
    """
    robot = env.scene["robot"]
    robot_pos_xy = robot.data.root_pos_w[:, :2]
    
    # Target zone bounds
    target_center = TARGET_ZONE_CENTER.to(robot_pos_xy.device)
    target_size = TARGET_ZONE_SIZE.to(robot_pos_xy.device)
    
    # Check if within target zone
    in_x = torch.abs(robot_pos_xy[:, 0] - target_center[0]) < target_size[0]
    in_y = torch.abs(robot_pos_xy[:, 1] - target_center[1]) < target_size[1]
    
    return in_x & in_y


def check_out_of_bounds(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if robot is out of the valid area.
    
    Returns:
        Boolean tensor (num_envs,): True if robot is out of bounds
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w
    
    # Define bounds (warehouse area)
    x_bounds = (-10.0, 10.0)
    y_bounds = (-10.0, 10.0)
    
    out_of_x = (robot_pos[:, 0] < x_bounds[0]) | (robot_pos[:, 0] > x_bounds[1])
    out_of_y = (robot_pos[:, 1] < y_bounds[0]) | (robot_pos[:, 1] > y_bounds[1])
    
    return out_of_x | out_of_y


def check_robot_orientation(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if robot orientation is too tilted (about to fall).
    
    Returns:
        Boolean tensor (num_envs,): True if robot is too tilted
    """
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
    
    # If w component is too small, robot is tilted too much
    # For upright, quat should be close to [1, 0, 0, 0]
    too_tilted = torch.abs(quat[:, 0]) < 0.5  # Less than 60 degrees from upright
    
    return too_tilted


__all__ = [
    "check_robot_fall",
    "check_reached_target",
    "check_out_of_bounds",
    "check_robot_orientation",
]
