import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.g1_tasks.move_artvip_livingroom_g1_29dof_dex3_wholebody import mdp
from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_scene.base_scene_artvip_livingroom_cfg import ArtVIPLivingroomSceneCfg

project_root = os.environ.get("PROJECT_ROOT")

ROBOT_INIT_POS = (5.0, 0.0, 0.8)
ROBOT_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
SCENE_DEACTIVATE_KEYWORDS = ("smallsofa",)
SCENE_DEACTIVATE_EXCLUDE_KEYWORDS = ()
LIVING_GROUP_OFFSET = (0.0, 0.0, 0.0)
SMALLSOFA_OFFSET = (0.0, 0.0, 0.0)
TEATABLE_OFFSET = (0.0, 0.0, 0.0)
OFFICECHAIR_OFFSET = (-1.0, -0.5, 0.0)
TABLE_DRINK_USD_PATH = f"{project_root}/assets/smalllivingroom/drink016/model_drink016.usd"
VASE_USD_PATH = f"{project_root}/assets/smalllivingroom/IKEA_PELARBJORK_vase_grey_green_20cm/model_PELARBJORK_vase_grey_green_20cm.usd"
CUSHION_USD_PATH = f"{project_root}/assets/smalllivingroom/IKEA_STOCKHOLM_2025_cushion_cover_brown_red_bright_pink_40x58cm/model_STOCKHOLM_2025_cushion_cover_brown_red_bright_pink_40x58cm.usd"
STOOL_USD_PATH = f"{project_root}/assets/smalllivingroom/stool/model_stool.usd"
BASKET_USD_PATH = f"{project_root}/assets/smalllivingroom/shopping basket007/basket.usd"
OBSTACLE_EASEL_USD_PATH = f"{project_root}/assets/smalllivingroom/obstacle/easel/model_easel_0.usd"
OBSTACLE_FLOOR_LAMP_USD_PATH = f"{project_root}/assets/smalllivingroom/obstacle/floor lamp003/model_floor_lamp_003.usd"
OBSTACLE_FOLDING_CHAIR_USD_PATH = f"{project_root}/assets/smalllivingroom/obstacle/folding chair002/model_chair_2.usd"
OBSTACLE_GLOBE_USD_PATH = f"{project_root}/assets/smalllivingroom/obstacle/globe/model_globe_0.usd"
OBSTACLE_PLATE_RACK_USD_PATH = f"{project_root}/assets/smalllivingroom/obstacle/plate rack002/platerack.usd"
OBSTACLE_CARTON_USD_PATH = f"{project_root}/assets/smalllivingroom/obstacle/carton/model_carton.usd"
TABLE_DRINK_INIT_POS = (6.0, 4.3, 0.4)
TABLE_DRINK_INIT_ROT = (1.0, 0.0, 0.0, 0.0)
GROUND_OBSTACLE_1_POS = (5.95, 3.90, 0.14)
GROUND_OBSTACLE_1_ROT = (0.9848, 0.1736, 0.0, 0.0)
GROUND_OBSTACLE_2_POS = (5.85, 2.60, 0.10)
GROUND_OBSTACLE_2_ROT = (0.9659, -0.2588, 0.0, 0.0)
GROUND_OBSTACLE_3_POS = (6.65, 2.05, 0.12)
GROUND_OBSTACLE_3_ROT = (0.9914, 0.1305, 0.0, 0.0)
GROUND_OBSTACLE_4_POS = (6.28, 1.45, 0.12)
GROUND_OBSTACLE_4_ROT = (0.9962, 0.0872, 0.0, 0.0)
GROUND_OBSTACLE_5_POS = (5.0, 3.85, 0.12)
GROUND_OBSTACLE_5_ROT = (0.9763, -0.2164, 0.0, 0.0)
GROUND_OBSTACLE_6_POS = (5.42, 2.75, 0.12)
GROUND_OBSTACLE_6_ROT = (0.9537, 0.3007, 0.0, 0.0)
GROUND_OBSTACLE_7_POS = (7.0, 1.55, 0.10)
GROUND_OBSTACLE_7_ROT = (0.9888, -0.1494, 0.0, 0.0)
GROUND_OBSTACLE_8_POS = (5.35, 1.05, 0.12)
GROUND_OBSTACLE_8_ROT = (0.9613, 0.2756, 0.0, 0.0)
GROUND_OBSTACLE_9_POS = (5.95, 1.55, 0.10)
GROUND_OBSTACLE_9_ROT = (0.6644, -0.2418, -0.2418, 0.6644)
GROUND_OBSTACLE_10_POS = (6.60, 1.0, 0.10)
GROUND_OBSTACLE_10_ROT = (0.9537, 0.3007, 0.0, 0.0)
TABLE_DRINK_MASS = 0.35
GROUND_OBSTACLE_1_MASS = 1.2
GROUND_OBSTACLE_2_MASS = 0.8
GROUND_OBSTACLE_3_MASS = 1.8
GROUND_OBSTACLE_4_MASS = 1.5
GROUND_OBSTACLE_5_MASS = 2.8
GROUND_OBSTACLE_6_MASS = 1.4
GROUND_OBSTACLE_7_MASS = 1.3
GROUND_OBSTACLE_8_MASS = 1.0
GROUND_OBSTACLE_9_MASS = 2.4
GROUND_OBSTACLE_10_MASS = 1.4

def _sum_offset(base, delta):
    return (
        float(base[0]) + float(delta[0]),
        float(base[1]) + float(delta[1]),
        float(base[2]) + float(delta[2]),
    )


SCENE_REPOSITION_RULES = (
    {
        "name": "smallsofa",
        "keywords": ("smallsofa",),
        "offset": _sum_offset(LIVING_GROUP_OFFSET, SMALLSOFA_OFFSET),
    },
    {
        "name": "teatable",
        "keywords": ("teatable",),
        "offset": _sum_offset(LIVING_GROUP_OFFSET, TEATABLE_OFFSET),
    },
    {
        "name": "officechair",
        "keywords": ("officechair_3", "officechair", "office chair"),
        "offset": OFFICECHAIR_OFFSET,
    },
)


@configclass
class ArtVIPLivingroomNoSofaTerrainSceneCfg(ArtVIPLivingroomSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_wholebody(
        init_pos=ROBOT_INIT_POS,
        init_rot=ROBOT_INIT_ROT,
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.g1_front_camera()
    world_camera = CameraPresets.g1_world_camera()

    table_drink = AssetBaseCfg(
        prim_path="/World/envs/env_.*/TableDrink",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(TABLE_DRINK_INIT_POS),
            rot=list(TABLE_DRINK_INIT_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=TABLE_DRINK_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=TABLE_DRINK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_1 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink1",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_1_POS),
            rot=list(GROUND_OBSTACLE_1_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=VASE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_1_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_2_POS),
            rot=list(GROUND_OBSTACLE_2_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=CUSHION_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_2_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_3 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink3",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_3_POS),
            rot=list(GROUND_OBSTACLE_3_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=STOOL_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_3_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_4 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink4",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_4_POS),
            rot=list(GROUND_OBSTACLE_4_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=BASKET_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_4_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_5 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink5",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_5_POS),
            rot=list(GROUND_OBSTACLE_5_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_EASEL_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_5_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_6 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink6",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_6_POS),
            rot=list(GROUND_OBSTACLE_6_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_GLOBE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_6_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_7 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink7",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_7_POS),
            rot=list(GROUND_OBSTACLE_7_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_PLATE_RACK_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_7_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_8 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink8",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_8_POS),
            rot=list(GROUND_OBSTACLE_8_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_CARTON_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_8_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_9 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink9",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_9_POS),
            rot=list(GROUND_OBSTACLE_9_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_FLOOR_LAMP_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_9_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    floor_obstacle_drink_10 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/FloorObstacleDrink10",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=list(GROUND_OBSTACLE_10_POS),
            rot=list(GROUND_OBSTACLE_10_ROT),
        ),
        spawn=UsdFileCfg(
            usd_path=OBSTACLE_GLOBE_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=GROUND_OBSTACLE_10_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.01,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward_livingroom, weight=1.0)


@configclass
class EventCfg:
    pass


@configclass
class MoveArtVIPLivingroomNoSofaG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: ArtVIPLivingroomNoSofaTerrainSceneCfg = ArtVIPLivingroomNoSofaTerrainSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    curriculum = None

    def __post_init__(self):
        self.scene_deactivate_keywords = SCENE_DEACTIVATE_KEYWORDS
        self.scene_deactivate_exclude_keywords = SCENE_DEACTIVATE_EXCLUDE_KEYWORDS
        self.scene_reposition_rules = SCENE_REPOSITION_RULES
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"
