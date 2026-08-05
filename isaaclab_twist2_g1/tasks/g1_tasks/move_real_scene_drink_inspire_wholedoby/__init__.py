import gymnasium as gym

from . import move_real_scene_drink_inspire_hw_env_cfg


gym.register(
    id="Isaac-Move-Real-Scene-Drink-G129-Inspire-Wholedoby",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_real_scene_drink_inspire_hw_env_cfg.MoveRealSceneDrinkG129InspireWholedobyEnvCfg,
    },
    disable_env_checker=True,
)
