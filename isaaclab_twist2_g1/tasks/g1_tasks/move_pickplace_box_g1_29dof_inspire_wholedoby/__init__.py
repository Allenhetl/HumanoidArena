import gymnasium as gym

from . import move_pickplace_box_g1_29dof_inspire_hw_env_cfg


gym.register(
    id="Isaac-Move-PickPlace-Box-G129-Inspire-Wholedoby",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_pickplace_box_g1_29dof_inspire_hw_env_cfg.MovePickPlaceBoxG129InspireWholedobyEnvCfg,
    },
    disable_env_checker=True,
)
