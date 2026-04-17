from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TARGET_REACHED_RADIUS_M = 0.75
DISTANCE_REWARD_SCALE_M = 1.5
SUCCESS_BONUS = 2.0


def _candidate_target_paths(env_idx: int) -> list[str]:
    base = f"/World/envs/env_{env_idx}/TargetSign"
    return [base, f"{base}/PRootNode"]


def _read_prim_world_position(path: str) -> tuple[float, float, float] | None:
    try:
        import omni.usd
        from pxr import Usd, UsdGeom
    except Exception:
        return None

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return None

    prim = stage.GetPrimAtPath(path)
    if prim is None or not prim.IsValid() or not prim.IsActive():
        return None

    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        return (float(translation[0]), float(translation[1]), float(translation[2]))
    except Exception:
        return None


def _get_target_positions(env: "ManagerBasedRLEnv") -> list[tuple[float, float, float] | None]:
    positions: list[tuple[float, float, float] | None] = []
    for env_idx in range(env.num_envs):
        target_position = None
        for path in _candidate_target_paths(env_idx):
            target_position = _read_prim_world_position(path)
            if target_position is not None:
                break
        positions.append(target_position)
    return positions


def compute_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    reward = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    robot_root_state = env.scene["robot"].data.root_state_w[:, 0:3]
    target_positions = _get_target_positions(env)

    for env_idx, target_position in enumerate(target_positions):
        if target_position is None:
            continue
        target_xy = torch.tensor(
            [target_position[0], target_position[1]],
            device=env.device,
            dtype=robot_root_state.dtype,
        )
        distance_xy = torch.linalg.vector_norm(robot_root_state[env_idx, 0:2] - target_xy, ord=2)
        reward[env_idx] = torch.exp(-distance_xy / DISTANCE_REWARD_SCALE_M)
        if bool(distance_xy <= TARGET_REACHED_RADIUS_M):
            reward[env_idx] += SUCCESS_BONUS

    return reward


__all__ = ["compute_reward"]
