from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def compute_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
