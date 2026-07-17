import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg

project_root = os.environ.get('PROJECT_ROOT')

DEFAULT_ROOM_USD_PATH = (
    f'{project_root}/assets/objects/real_scene/'
    'small_warehouse_digital_twin_office_fixed_gauss_xneg90_ccm1.usda'
)
ROOM_USD_PATH = os.environ.get('REAL_SCENE_ROOM_USD', DEFAULT_ROOM_USD_PATH)


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

    ambient_light = AssetBaseCfg(
        prim_path='/World/ambient_light',
        spawn=sim_utils.DomeLightCfg(
            color=(0.8, 0.8, 0.8),
            intensity=1200.0,
        ),
    )

    robot_key_light = AssetBaseCfg(
        prim_path='/World/robot_key_light',
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.5, 4.0, 1.4),
        ),
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 0.96, 0.9),
            intensity=10000.0,
            radius=1.0,
        ),
    )

    world_camera = CameraBaseCfg.get_world_camera_config(
        pos_offset=(1.5, 2.85, 1.25),
        rot_offset=(0.85749, 0.51450, 0.0, 0.0),
        focal_length=12,
        horizontal_aperture=27,
        convention='opengl',
    )
