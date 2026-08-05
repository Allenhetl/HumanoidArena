from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


MIN_ROOT_HEIGHT = 0.35


def fall_detected(env: "ManagerBasedRLEnv"):
    root_height = env.scene["robot"].data.root_state_w[:, 2]
    return root_height < MIN_ROOT_HEIGHT
