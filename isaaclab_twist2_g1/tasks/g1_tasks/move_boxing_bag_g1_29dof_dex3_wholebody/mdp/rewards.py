# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Minimal reward for boxing bag task (主要用途：遙操作數據採集)."""

import torch

from isaaclab.envs import ManagerBasedRLEnv


def compute_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Placeholder reward. Task focus: RGB + robot action data collection."""
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


__all__ = ["compute_reward"]
