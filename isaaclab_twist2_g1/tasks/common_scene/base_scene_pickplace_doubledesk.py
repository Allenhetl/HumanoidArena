import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import   CameraBaseCfg  # isort: skip
import os
project_root = os.environ.get("PROJECT_ROOT")
PACKING_TABLE_L_POS = [-0.1, -3.2, -0.2]
PACKING_TABLE_R_POS = [-4.0, -3.2, -0.2]
OBJECT_L_POS_OFFSET = [-0.3, 0.0, 1.04]
CONTAINER_R_POS_OFFSET = [0.3, 0.0, 1.00]
@configclass
class DoubleTableSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
    # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    room_walls = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Room",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0],  # room center point
            rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_doubledesk.usd",
        ),
    )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )
    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-1.9, -5.0, 1.8),
                                                    rot_offset=( -0.40614,0.78544, 0.4277, -0.16986))
