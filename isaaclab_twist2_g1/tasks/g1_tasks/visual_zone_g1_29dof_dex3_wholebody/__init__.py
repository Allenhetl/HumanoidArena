# Copyright (c) 2025, HumanoidArena Project
# License: Apache License, Version 2.0

"""Visual zone navigation task for G1 robot with Dex3 hands."""

import gymnasium as gym

from . import visual_zone_g1_29dof_dex3_hw_env_cfg


gym.register(
    id="Isaac-Visual-Zone-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": visual_zone_g1_29dof_dex3_hw_env_cfg.VisualZoneG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
