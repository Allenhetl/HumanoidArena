import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from tasks.common_config import CameraBaseCfg

project_root = os.environ.get("PROJECT_ROOT")

PILE_RADIUS = 0.12
PILE_HEIGHT = 0.20
PILE_POSITIONS = [
    (0.0, -0.8),
    (0.4, -0.3),
    (-0.4, -0.2),
    (0.2, 0.3),
    (-0.2, 0.5),
    (0.5, 0.9),
    (-0.5, 1.1),
    (0.0, 1.5),
]


def _plum_pile_cfg(name: str, x: float, y: float) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[x, y, PILE_HEIGHT * 0.5],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=PILE_RADIUS,
            height=PILE_HEIGHT,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.35, 0.35, 0.35),
                metallic=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10.0,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )


@configclass
class PlumBlossomPileSceneCfg(InteractiveSceneCfg):
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd",
        ),
    )

    pile_1 = _plum_pile_cfg("PlumPile1", *PILE_POSITIONS[0])
    pile_2 = _plum_pile_cfg("PlumPile2", *PILE_POSITIONS[1])
    pile_3 = _plum_pile_cfg("PlumPile3", *PILE_POSITIONS[2])
    pile_4 = _plum_pile_cfg("PlumPile4", *PILE_POSITIONS[3])
    pile_5 = _plum_pile_cfg("PlumPile5", *PILE_POSITIONS[4])
    pile_6 = _plum_pile_cfg("PlumPile6", *PILE_POSITIONS[5])
    pile_7 = _plum_pile_cfg("PlumPile7", *PILE_POSITIONS[6])
    pile_8 = _plum_pile_cfg("PlumPile8", *PILE_POSITIONS[7])

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-1.9, -5.0, 1.8),
        rot_offset=(-0.40614, 0.78544, 0.4277, -0.16986),
    )
