# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import move_boxing_bag_g1_29dof_dex3_hw_env_cfg
from . import move_boxing_bag_hanging_hw_env_cfg

gym.register(
    id="Isaac-Move-Boxing-Bag-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_boxing_bag_g1_29dof_dex3_hw_env_cfg.MoveBoxingBagG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Move-Boxing-Bag-Hanging-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_boxing_bag_hanging_hw_env_cfg.MoveBoxingBagHangingG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
