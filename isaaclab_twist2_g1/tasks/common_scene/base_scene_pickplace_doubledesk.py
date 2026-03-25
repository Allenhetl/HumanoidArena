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
            usd_path=f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd",
        ),
    )

    # 2. table configuration
    packing_table_l = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_l",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=PACKING_TABLE_L_POS,   # initial position [x, y, z]
                                                rot=[0.7071, 0.0, 0.0, 0.7071]), # initial rotation [x, y, z, w]
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",    # table model file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )
    packing_table_r = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable_r",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=PACKING_TABLE_R_POS,   # initial position [x, y, z]
                                                rot=[-0.7071, 0.0, 0.0, 0.7071]), # initial rotation [x, y, z, w]
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",    # table model file
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
        ),
    )

    # 3. object configuration
    object_l = RigidObjectCfg(
        prim_path="/World/envs/env_.*/_l",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[
                PACKING_TABLE_L_POS[0] + OBJECT_L_POS_OFFSET[0],
                PACKING_TABLE_L_POS[1] + OBJECT_L_POS_OFFSET[1],
                PACKING_TABLE_L_POS[2] + OBJECT_L_POS_OFFSET[2],
            ],
            rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.06,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), metallic=0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
        ),
    )

    container_r = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Container_r",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[
                PACKING_TABLE_R_POS[0] + CONTAINER_R_POS_OFFSET[0],
                PACKING_TABLE_R_POS[1] + CONTAINER_R_POS_OFFSET[1],
                PACKING_TABLE_R_POS[2] + CONTAINER_R_POS_OFFSET[2],
            ],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/Props/general/SM_Crate_A08_Blue_01/SM_Crate_A08_Blue_01_physics.usd",
            scale=(0.01, 0.01, 0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )
    
    #hurdle between the two tables
    hurdle = AssetBaseCfg(
        prim_path = "/World/envs/env_.*/hurdle",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-2.6, -3.2, 0.0],
            rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 5.0, 0.10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.15, 0.05), metallic=0
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                restitution_combine_mode="min",
                static_friction=10,
                dynamic_friction=1.5,
                restitution=0.01,
            ),
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
