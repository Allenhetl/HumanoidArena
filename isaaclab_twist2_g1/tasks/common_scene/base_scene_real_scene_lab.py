import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg

_project_root = os.environ.get('PROJECT_ROOT', '')

# Default paths/poses for the original small_warehouse scene.
# These can be overridden via YAML (scene.room.spawn.usd_path,
# scene.robot.init_state.pos, scene.robot.init_state.rot) or via
# environment variables as a fallback.
_DEFAULT_ROOM_USD_PATH = os.path.join(
    _project_root, 'assets', 'objects', 'real_scene',
    'small_warehouse_digital_twin_office_fixed_gauss_xneg90_ccm1.usda',
)
_DEFAULT_ROBOT_INIT_POS = (1.5, 4.0, -0.28)
# quaternion order: (w, x, y, z)
# default (1, 0, 0, 0) = identity, robot faces -X
# rotate 180 deg around Z: (0, 0, 0, 1)
# rotate 90 deg around Z: (0.7071, 0, 0, 0.7071)
_DEFAULT_ROBOT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


def _resolve_room_usd_path():
    return os.environ.get('REAL_SCENE_ROOM_USD', _DEFAULT_ROOM_USD_PATH)


def _resolve_robot_init_pos():
    env_val = os.environ.get('REAL_SCENE_ROBOT_INIT_POS')
    if not env_val:
        return _DEFAULT_ROBOT_INIT_POS
    parts = [float(v.strip()) for v in env_val.split(',')]
    if len(parts) != 3:
        raise ValueError(
            f'REAL_SCENE_ROBOT_INIT_POS must have 3 comma-separated values, got: {env_val}'
        )
    return (parts[0], parts[1], parts[2])


def _resolve_robot_init_rot():
    env_val = os.environ.get('REAL_SCENE_ROBOT_INIT_ROT')
    if not env_val:
        return _DEFAULT_ROBOT_INIT_ROT
    parts = [float(v.strip()) for v in env_val.split(',')]
    if len(parts) != 4:
        raise ValueError(
            f'REAL_SCENE_ROBOT_INIT_ROT must have 4 comma-separated values (w,x,y,z), got: {env_val}'
        )
    return (parts[0], parts[1], parts[2], parts[3])


@configclass
class RealSceneLabSceneCfg(InteractiveSceneCfg):
    # ------------------------------------------------------------------
    # Scene assets
    # ------------------------------------------------------------------
    # The room USD path and robot init pose are resolved at class-definition
    # time from env vars (fallback: small_warehouse defaults).  YAML overrides
    # applied after parse_env_cfg can patch scene.room.spawn.usd_path,
    # scene.robot.init_state.pos, scene.robot.init_state.rot, and
    # scene.robot_key_light.init_state.pos before gym.make triggers env.reset().
    room = AssetBaseCfg(
        prim_path='/World/envs/env_.*/Room',
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=_resolve_room_usd_path(),
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
            pos=(_resolve_robot_init_pos()[0], _resolve_robot_init_pos()[1], 1.4),
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
