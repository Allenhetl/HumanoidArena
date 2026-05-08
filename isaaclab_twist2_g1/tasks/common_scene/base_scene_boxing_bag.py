import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass


project_root = os.environ.get("PROJECT_ROOT")

ROOM_USD_PATH = (
    f"{project_root}/assets/objects/small_warehouse/"
    "small_warehouse_boxing/boxing_scene.usd"
)
BOXING_TARGET_USD_PATH = (
    f"{project_root}/assets/objects/small_warehouse/small_warehouse_boxing/"
    "interaction_obj/boxing_target.usd"
)

# Startup pose only. Reset-time randomization is controlled by the task cfg and YAML.
BOXING_TARGET_INIT_POS = (0.0, 0.0, 0.0)
BOXING_TARGET_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


@configclass
class BoxingBagSceneCfg(InteractiveSceneCfg):
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=ROOM_USD_PATH,
        ),
    )

    boxing_target = AssetBaseCfg(
        prim_path="/World/envs/env_.*/BoxingTarget",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BOXING_TARGET_INIT_POS,
            rot=BOXING_TARGET_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=BOXING_TARGET_USD_PATH,
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )
