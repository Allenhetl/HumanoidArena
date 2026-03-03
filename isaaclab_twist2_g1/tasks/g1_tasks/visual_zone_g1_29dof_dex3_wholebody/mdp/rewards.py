# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0

"""Reward functions for visual zone navigation task."""

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Zone definitions (from scene configuration)
TARGET_ZONE_CENTER = torch.tensor([0.0, 3.0])
TARGET_ZONE_SIZE = torch.tensor([1.0, 1.0])  # half-size in x, y

FORBIDDEN_ZONE_CENTER = torch.tensor([1.5, 2.9])
FORBIDDEN_ZONE_SIZE = torch.tensor([0.4, 0.4])  # half-size in x, y

ROBOT_FALL_HEIGHT_THRESHOLD = 0.3  # meters


def _is_in_zone(pos_xy: torch.Tensor, zone_center: torch.Tensor, zone_size: torch.Tensor) -> torch.Tensor:
    """Check if positions are inside a rectangular zone.
    
    Args:
        pos_xy: Robot positions (num_envs, 2)
        zone_center: Zone center (2,)
        zone_size: Zone half-size (2,)
    
    Returns:
        Boolean tensor (num_envs,)
    """
    zone_center = zone_center.to(pos_xy.device)
    zone_size = zone_size.to(pos_xy.device)
    
    # Check if within bounds
    in_x = torch.abs(pos_xy[:, 0] - zone_center[0]) < zone_size[0]
    in_y = torch.abs(pos_xy[:, 1] - zone_center[1]) < zone_size[1]
    
    return in_x & in_y


def reward_reach_target_zone(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for reaching the target zone (green area).
    
    Returns:
        Tensor (num_envs,): 1.0 if in target zone, 0.0 otherwise
    """
    robot = env.scene["robot"]
    robot_pos_xy = robot.data.root_pos_w[:, :2]
    
    in_target = _is_in_zone(robot_pos_xy, TARGET_ZONE_CENTER, TARGET_ZONE_SIZE)
    
    return in_target.float()


def reward_avoid_forbidden_zone(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for entering the forbidden zone (red area).
    
    Returns:
        Tensor (num_envs,): 1.0 if in forbidden zone (to be weighted negatively), 0.0 otherwise
    """
    robot = env.scene["robot"]
    robot_pos_xy = robot.data.root_pos_w[:, :2]
    
    in_forbidden = _is_in_zone(robot_pos_xy, FORBIDDEN_ZONE_CENTER, FORBIDDEN_ZONE_SIZE)
    
    return in_forbidden.float()


def reward_move_towards_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for moving towards the target zone.
    
    Uses velocity in the direction of the target.
    
    Returns:
        Tensor (num_envs,): reward based on velocity towards target
    """
    robot = env.scene["robot"]
    robot_pos_xy = robot.data.root_pos_w[:, :2]
    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]
    
    # Direction to target
    target_center = TARGET_ZONE_CENTER.to(robot_pos_xy.device)
    direction = target_center - robot_pos_xy
    
    # Normalize direction
    distance = torch.norm(direction, dim=-1, keepdim=True).clamp(min=1e-6)
    direction_normalized = direction / distance
    
    # Velocity component in target direction
    vel_towards_target = torch.sum(robot_vel_xy * direction_normalized, dim=-1)
    
    # Positive reward for moving towards target
    reward = torch.clamp(vel_towards_target, min=0.0, max=1.0)
    
    return reward


def penalty_robot_fall(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for robot falling (base height too low).
    
    Returns:
        Tensor (num_envs,): 1.0 if fallen (to be weighted negatively), 0.0 otherwise
    """
    robot = env.scene["robot"]
    robot_height = robot.data.root_pos_w[:, 2]
    
    has_fallen = robot_height < ROBOT_FALL_HEIGHT_THRESHOLD
    
    return has_fallen.float()


def penalty_action_magnitude(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalty for large actions (energy efficiency).
    
    Returns:
        Tensor (num_envs,): sum of squared action magnitudes
    """
    # Get actions from action manager
    action = env.action_manager.action
    
    if action is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Sum of squared actions
    action_penalty = torch.sum(action ** 2, dim=-1)
    
    return action_penalty


def reward_distance_to_target(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Negative reward based on distance to target (closer is better).
    
    Returns:
        Tensor (num_envs,): negative distance (to be weighted positively for closer = better)
    """
    robot = env.scene["robot"]
    robot_pos_xy = robot.data.root_pos_w[:, :2]
    
    target_center = TARGET_ZONE_CENTER.to(robot_pos_xy.device)
    distance = torch.norm(robot_pos_xy - target_center, dim=-1)
    
    # Normalize and invert: closer = higher reward
    # Max expected distance is about 5 meters
    reward = 1.0 - torch.clamp(distance / 5.0, max=1.0)
    
    return reward


def reward_stable_standing(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for maintaining stable standing posture.
    
    Based on base orientation being close to upright.
    
    Returns:
        Tensor (num_envs,): reward for upright posture
    """
    robot = env.scene["robot"]
    quat = robot.data.root_quat_w  # (num_envs, 4) [w, x, y, z]
    
    # For upright posture, quat should be close to [1, 0, 0, 0]
    # We check the w component (should be close to 1 for upright)
    upright_reward = torch.abs(quat[:, 0])  # w component
    
    return upright_reward


__all__ = [
    "reward_reach_target_zone",
    "reward_avoid_forbidden_zone",
    "reward_move_towards_target",
    "penalty_robot_fall",
    "penalty_action_magnitude",
    "reward_distance_to_target",
    "reward_stable_standing",
]
