import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg


project_root = os.environ.get("PROJECT_ROOT")
INTERACTION_OBJ_ROOT = (
    f"{project_root}/assets/objects/small_warehouse/"
    "small_warehouse_grapcup/interaction_obj"
)

ROOM_USD_PATH = f"{project_root}/assets/objects/small_warehouse/small_warehouse_grapcup/small_warehouse_digital_twin_grapcup.usd"
DRINK016_USD_PATH = f"{INTERACTION_OBJ_ROOT}/drink016/model_drink016.usd"
DRINK_INIT_POS = (-0.18, 0.5, 0.82)
DRINK_INIT_ROT = (0.78, 0.0, 0.0, -0.61)

CHAIR001_USD_PATH = f"{INTERACTION_OBJ_ROOT}/chair001/model_chair1.usd"
PLATE_RACK_USD_PATH = f"{INTERACTION_OBJ_ROOT}/plate rack002/platerack.usd"
VASE_USD_PATH = (
    f"{INTERACTION_OBJ_ROOT}/IKEA_PELARBJORK_vase_grey_green_20cm/"
    "model_PELARBJORK_vase_grey_green_20cm.usd"
)
BOOK17_USD_PATH = f"{INTERACTION_OBJ_ROOT}/book_17/model_book_17.usd"

OBSTACLE_01_INIT_POS = (-2.45, -1.5, 0.0)
OBSTACLE_01_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
OBSTACLE_02_INIT_POS = (-1.65, -1.4, 0.0)
OBSTACLE_02_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
OBSTACLE_03_INIT_POS = (-3.325, -0.75, 0.0)
OBSTACLE_03_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
OBSTACLE_04_INIT_POS = (-3.075, -1.32, 0.0)
OBSTACLE_04_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
OBSTACLE_05_INIT_POS = (-2.15, -0.95, 0.0)
OBSTACLE_05_INIT_ROT = (1.0, 0.0, 0.0, 0.0)


@configclass
class LivingroomGrapCupSceneCfg(InteractiveSceneCfg):
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

    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=DRINK_INIT_POS,
            rot=DRINK_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=DRINK016_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.7),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.1,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    obstacle_01 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle01",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_01_INIT_POS,
            rot=OBSTACLE_01_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=CHAIR001_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    obstacle_02 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle02",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_02_INIT_POS,
            rot=OBSTACLE_02_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=PLATE_RACK_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.2),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    obstacle_03 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle03",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_03_INIT_POS,
            rot=OBSTACLE_03_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=VASE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    obstacle_04 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle04",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_04_INIT_POS,
            rot=OBSTACLE_04_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=BOOK17_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    obstacle_05 = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Obstacle05",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=OBSTACLE_05_INIT_POS,
            rot=OBSTACLE_05_INIT_ROT,
        ),
        spawn=UsdFileCfg(
            usd_path=CHAIR001_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-0.1, 3.6, 1.6),
        rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
        focal_length=16.5,
    )
