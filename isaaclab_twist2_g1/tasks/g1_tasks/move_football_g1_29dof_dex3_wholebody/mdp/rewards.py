# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from tasks.common_scene.base_scene_football_single_cfg_wholebody import GOAL_NET_1_ORIGIN, GOAL_NET_2_ORIGIN

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_reward(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    front_goal_origin_x: float = GOAL_NET_1_ORIGIN[0],
    front_goal_origin_y: float = GOAL_NET_1_ORIGIN[1],
    back_goal_origin_x: float = GOAL_NET_2_ORIGIN[0],
    back_goal_origin_y: float = GOAL_NET_2_ORIGIN[1],
    goal_local_x_min: float = -0.05,
    goal_local_x_max: float = 5.05,
    goal_local_y_min: float = -0.05,
    goal_local_y_max: float = 2.10,
    goal_min_height: float = 0.00,
    goal_max_height: float = 1.85,
    ball_radius: float = 0.11,
) -> torch.Tensor:
    """Binary football reward based on the goal asset bounds.

    The single-goal task uses only ``goal_net``. This reward therefore always
    checks the front goal and only checks the mirrored back goal when the scene
    actually contains ``goal_net_2``.
    """

    football: RigidObject = env.scene[object_cfg.name]

    ball_x = football.data.root_pos_w[:, 0]
    ball_y = football.data.root_pos_w[:, 1]
    ball_z = football.data.root_pos_w[:, 2]

    # These local bounds come from the football goal asset's XY footprint after
    # the 0.01 scale in the scene config is applied, with a small safety margin.
    front_x_min = front_goal_origin_x + goal_local_x_min - ball_radius
    front_x_max = front_goal_origin_x + goal_local_x_max + ball_radius
    front_y_min = front_goal_origin_y + goal_local_y_min - ball_radius
    front_y_max = front_goal_origin_y + goal_local_y_max + ball_radius

    in_goal_z = (ball_z > goal_min_height - ball_radius) & (ball_z < goal_max_height + ball_radius)

    in_front_goal_x = (ball_x > front_x_min) & (ball_x < front_x_max)
    in_front_goal_y = (ball_y > front_y_min) & (ball_y < front_y_max)
    scored_front = in_front_goal_x & in_front_goal_y & in_goal_z

    has_back_goal = "goal_net_2" in env.scene.keys()
    if has_back_goal:
        back_x_min = back_goal_origin_x - goal_local_x_max - ball_radius
        back_x_max = back_goal_origin_x - goal_local_x_min + ball_radius
        back_y_min = back_goal_origin_y - goal_local_y_max - ball_radius
        back_y_max = back_goal_origin_y - goal_local_y_min + ball_radius
        in_back_goal_x = (ball_x > back_x_min) & (ball_x < back_x_max)
        in_back_goal_y = (ball_y > back_y_min) & (ball_y < back_y_max)
        scored_back = in_back_goal_x & in_back_goal_y & in_goal_z
    else:
        scored_back = torch.zeros_like(scored_front)

    scored = scored_front | scored_back

    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    reward[~scored] = -1.0
    reward[scored] = 1.0
    return reward


__all__ = ["compute_reward"]
