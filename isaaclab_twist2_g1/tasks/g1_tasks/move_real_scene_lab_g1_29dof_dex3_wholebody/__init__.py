import gymnasium as gym

from . import move_real_scene_lab_g1_29dof_dex3_hw_env_cfg


gym.register(
    id="Isaac-Move-Real-Scene-Lab-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_real_scene_lab_g1_29dof_dex3_hw_env_cfg.MoveRealSceneLabG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
