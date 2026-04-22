import torch

from isaaclab.envs import ManagerBasedRLEnv


def compute_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Placeholder reward. Task focus: teleoperation data collection."""
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


__all__ = ["compute_reward"]
