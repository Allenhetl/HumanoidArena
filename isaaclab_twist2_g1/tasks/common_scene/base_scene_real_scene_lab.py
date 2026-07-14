import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg

project_root = os.environ.get('PROJECT_ROOT')

ROOM_USD_PATH = (
    f'{project_root}/assets/objects/real_scene/'
    'small_warehouse_digital_twin_office.usdz'
)


@configclass
class RealSceneLabSceneCfg(InteractiveSceneCfg):
    room = AssetBaseCfg(
        prim_path='/World/envs/env_.*/Room',
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=ROOM_USD_PATH,
        ),
    )

    light = AssetBaseCfg(
        prim_path='/World/light',
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(3.0, 3.0, 2.5),
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 0.95, 0.85),
            intensity=8000.0,
            radius=0.3,
        ),
    )

    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(3.0, -2.0, 2.5),
        rot_offset=(0.5, 0.3, 0.4, 0.7),
        focal_length=12,
        horizontal_aperture=27,
        convention='opengl',
    )
