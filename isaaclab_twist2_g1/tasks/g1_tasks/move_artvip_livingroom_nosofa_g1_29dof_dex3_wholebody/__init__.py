import gymnasium as gym

from . import move_artvip_livingroom_nosofa_g1_29dof_dex3_hw_env_cfg


gym.register(
    id="Isaac-Move-ArtVIP-Livingroom-GrapCup-G129-Dex3-Wholebody",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": move_artvip_livingroom_nosofa_g1_29dof_dex3_hw_env_cfg.MoveArtVIPLivingroomNoSofaG129Dex3WholebodyEnvCfg,
    },
    disable_env_checker=True,
)
