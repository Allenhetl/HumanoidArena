import os

import torch

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.g1_tasks.move_artvip_livingroom_g1_29dof_dex3_wholebody import mdp
from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_runtime import (
    apply_optional_runtime_augments,
    apply_scene_filter_from_cfg,
    apply_scene_reposition_from_cfg,
)
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


def _reset_grapcup_task_objects_on_stage(env) -> int:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    pose_layout = [
        ("TableDrink", TABLE_DRINK_INIT_POS, TABLE_DRINK_INIT_ROT),
        ("FloorObstacleDrink1", GROUND_OBSTACLE_1_POS, GROUND_OBSTACLE_1_ROT),
        ("FloorObstacleDrink2", GROUND_OBSTACLE_2_POS, GROUND_OBSTACLE_2_ROT),
        ("FloorObstacleDrink3", GROUND_OBSTACLE_3_POS, GROUND_OBSTACLE_3_ROT),
        ("FloorObstacleDrink4", GROUND_OBSTACLE_4_POS, GROUND_OBSTACLE_4_ROT),
        ("FloorObstacleDrink5", GROUND_OBSTACLE_5_POS, GROUND_OBSTACLE_5_ROT),
        ("FloorObstacleDrink6", GROUND_OBSTACLE_6_POS, GROUND_OBSTACLE_6_ROT),
        ("FloorObstacleDrink7", GROUND_OBSTACLE_7_POS, GROUND_OBSTACLE_7_ROT),
        ("FloorObstacleDrink8", GROUND_OBSTACLE_8_POS, GROUND_OBSTACLE_8_ROT),
        ("FloorObstacleDrink9", GROUND_OBSTACLE_9_POS, GROUND_OBSTACLE_9_ROT),
        ("FloorObstacleDrink10", GROUND_OBSTACLE_10_POS, GROUND_OBSTACLE_10_ROT),
    ]

    def _set_pose(prim_path: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> bool:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or (not prim.IsActive()):
            return False
        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
                orient_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp()
        translate_op.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        orient_op.Set(Gf.Quatd(float(rot[0]), Gf.Vec3d(float(rot[1]), float(rot[2]), float(rot[3]))))
        return True

    reset_count = 0
    num_envs = int(getattr(env, "num_envs", 1))
    for env_idx in range(num_envs):
        prefix = f"/World/envs/env_{env_idx}"
        for prim_name, pos, rot in pose_layout:
            if _set_pose(f"{prefix}/{prim_name}", pos, rot):
                reset_count += 1
    return reset_count


def _capture_grapcup_rigid_local_transforms(env) -> dict[str, object]:
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    saved_transforms: dict[str, object] = {}
    num_envs = int(getattr(env, "num_envs", 1))
    object_names = [
        "TableDrink",
        "FloorObstacleDrink1",
        "FloorObstacleDrink2",
        "FloorObstacleDrink3",
        "FloorObstacleDrink4",
        "FloorObstacleDrink5",
        "FloorObstacleDrink6",
        "FloorObstacleDrink7",
        "FloorObstacleDrink8",
        "FloorObstacleDrink9",
        "FloorObstacleDrink10",
    ]
    for env_idx in range(num_envs):
        prefix = f"/World/envs/env_{env_idx}"
        for object_name in object_names:
            target_path = f"{prefix}/{object_name}"
            target_prim = stage.GetPrimAtPath(target_path)
            if not target_prim.IsValid():
                continue
            captured_any = False
            for sub_prim in Usd.PrimRange(target_prim):
                if not UsdPhysics.RigidBodyAPI.Get(stage, sub_prim.GetPath()):
                    continue
                xformable = UsdGeom.Xformable(sub_prim)
                local_tf = xformable.GetLocalTransformation()
                if isinstance(local_tf, tuple):
                    local_tf = local_tf[0]
                saved_transforms[str(sub_prim.GetPath())] = local_tf
                captured_any = True
            if not captured_any:
                xformable = UsdGeom.Xformable(target_prim)
                local_tf = xformable.GetLocalTransformation()
                if isinstance(local_tf, tuple):
                    local_tf = local_tf[0]
                saved_transforms[target_path] = local_tf
    return saved_transforms


def _restore_grapcup_rigid_local_transforms(saved_transforms: dict[str, object]) -> int:
    if not saved_transforms:
        return 0

    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    restored = 0
    for prim_path, local_tf in saved_transforms.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or (not prim.IsActive()):
            continue
        xformable = UsdGeom.Xformable(prim)
        ordered_ops = xformable.GetOrderedXformOps()
        if len(ordered_ops) == 1 and ordered_ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform:
            transform_op = ordered_ops[0]
        else:
            xformable.ClearXformOpOrder()
            transform_op = xformable.AddTransformOp()
        transform_op.Set(local_tf)
        restored += 1
    return restored

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

        self.event_manager = SimpleEventManager()
        self._task_adjust_args_cli = None
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(
                func=lambda env: self._reset_object_self(env)
            ),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: self._reset_all_self(env)
            ),
        )
        self._grapcup_saved_rigid_local_transforms = {}

    def initialize_task_scene(self, env, args_cli=None):
        self._task_adjust_args_cli = args_cli
        self.adjust_task_scene(env, phase="init", args_cli=args_cli)

    def adjust_task_scene(self, env, phase="init", args_cli=None):
        args_cli = args_cli if args_cli is not None else self._task_adjust_args_cli
        if phase == "init":
            self._apply_init_adjustments(env, args_cli=args_cli)
        elif phase == "reset":
            self._apply_reset_adjustments(env)

    def _reset_object_self(self, env):
        self.adjust_task_scene(env, phase="reset")

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        self.adjust_task_scene(env, phase="reset")

    def _apply_init_adjustments(self, env, args_cli=None):
        """Apply GrapCup-specific runtime stage patches before the initial reset.

        The original sim_main implementation ran these physics edits before
        ``env.sim.reset()`` / ``env.reset()`` so the spawned rigid bodies picked
        up the authored USD properties on first initialization.
        """
        apply_scene_filter_from_cfg(self)
        apply_scene_reposition_from_cfg(self)
        if args_cli is not None:
            apply_optional_runtime_augments(args_cli)
        try:
            import omni.usd
            from pxr import Usd, UsdGeom, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            obstacle_paths = [f"/World/envs/env_0/FloorObstacleDrink{i}" for i in range(1, 11)]
            target_paths = ["/World/envs/env_0/TableDrink", "/World/envs/env_0/Room/model_officechair_3"]
            target_paths += obstacle_paths

            def _enable_dynamic_collision(target_path: str) -> tuple[int, int, bool]:
                target_prim = stage.GetPrimAtPath(target_path)
                if not target_prim.IsValid():
                    return (0, 0, False)
                rigid_count = 0
                collider_count = 0
                has_existing_rigid = False
                for sub_prim in Usd.PrimRange(target_prim):
                    sub_rigid = UsdPhysics.RigidBodyAPI.Get(stage, sub_prim.GetPath())
                    if sub_rigid:
                        has_existing_rigid = True
                        sub_rigid.GetRigidBodyEnabledAttr().Set(True)
                        sub_rigid.GetKinematicEnabledAttr().Set(False)
                        rigid_count += 1
                    if sub_prim.IsA(UsdGeom.Mesh):
                        collision_api = UsdPhysics.CollisionAPI.Apply(sub_prim)
                        collision_api.GetCollisionEnabledAttr().Set(True)
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(sub_prim)
                        mesh_collision_api.GetApproximationAttr().Set("convexHull")
                        collider_count += 1
                if not has_existing_rigid:
                    root_rigid = UsdPhysics.RigidBodyAPI.Apply(target_prim)
                    root_rigid.GetRigidBodyEnabledAttr().Set(True)
                    root_rigid.GetKinematicEnabledAttr().Set(False)
                    rigid_count += 1
                return (rigid_count, collider_count, True)

            for target_path in target_paths:
                rigid_count, collider_count, ok = _enable_dynamic_collision(target_path)
                print(
                    f"[livingroom_collision] target={target_path} ok={ok} "
                    f"rigid_bodies={rigid_count} colliders={collider_count}"
                )

            desired_total_mass = {
                "/World/envs/env_0/TableDrink": 0.35,
                "/World/envs/env_0/Room/model_officechair_3": 8.0,
                "/World/envs/env_0/FloorObstacleDrink1": 1.2,
                "/World/envs/env_0/FloorObstacleDrink2": 0.8,
                "/World/envs/env_0/FloorObstacleDrink3": 1.8,
                "/World/envs/env_0/FloorObstacleDrink4": 1.5,
                "/World/envs/env_0/FloorObstacleDrink5": 2.8,
                "/World/envs/env_0/FloorObstacleDrink6": 1.4,
                "/World/envs/env_0/FloorObstacleDrink7": 1.3,
                "/World/envs/env_0/FloorObstacleDrink8": 1.0,
                "/World/envs/env_0/FloorObstacleDrink9": 2.4,
                "/World/envs/env_0/FloorObstacleDrink10": 1.4,
            }

            def _retune_mass(target_path: str) -> tuple[bool, int, float, float]:
                target_prim = stage.GetPrimAtPath(target_path)
                if not target_prim.IsValid():
                    return (False, 0, 0.0, 0.0)
                rigid_prims = []
                for sub_prim in Usd.PrimRange(target_prim):
                    if UsdPhysics.RigidBodyAPI.Get(stage, sub_prim.GetPath()):
                        rigid_prims.append(sub_prim)
                if not rigid_prims:
                    return (False, 0, 0.0, 0.0)
                target_total = float(desired_total_mass.get(target_path, 1.0))
                per_mass = max(0.02, target_total / float(len(rigid_prims)))
                for sub_prim in rigid_prims:
                    mass_api = UsdPhysics.MassAPI.Apply(sub_prim)
                    mass_api.GetMassAttr().Set(per_mass)
                return (True, len(rigid_prims), target_total, per_mass)

            for target_path in target_paths:
                ok_tune, rigid_cnt_tune, target_total, per_mass = _retune_mass(target_path)
                print(
                    f"[livingroom_mass_tune] target={target_path} ok={ok_tune} "
                    f"rigid_bodies={rigid_cnt_tune} target_total_mass={target_total} "
                    f"per_rigid_mass={per_mass}"
                )
            if not self._grapcup_saved_rigid_local_transforms:
                self._grapcup_saved_rigid_local_transforms = _capture_grapcup_rigid_local_transforms(env)
                print(
                    "[livingroom_runtime] captured rigid local transforms="
                    f"{len(self._grapcup_saved_rigid_local_transforms)}"
                )
        except Exception as exc:
            print(f"[livingroom_collision] setup failed: {exc}")

    def _apply_reset_adjustments(self, env):
        """Verify GrapCup post-reset stage state and emit optional debug probes."""
        apply_scene_filter_from_cfg(self)
        apply_scene_reposition_from_cfg(self)
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            reset_count = _reset_grapcup_task_objects_on_stage(env)
            restored_rigid_count = _restore_grapcup_rigid_local_transforms(
                self._grapcup_saved_rigid_local_transforms
            )
            verify_paths = [
                "/World/envs/env_0/TableDrink",
                "/World/envs/env_0/Room/model_officechair_3",
            ]
            verify_paths += [f"/World/envs/env_0/FloorObstacleDrink{i}" for i in range(1, 11)]
            valid_count = 0
            for target_path in verify_paths:
                prim = stage.GetPrimAtPath(target_path)
                if prim.IsValid() and prim.IsActive():
                    valid_count += 1
            print(
                f"[livingroom_post_reset] verified active task prims={valid_count}/{len(verify_paths)}, "
                f"reset_task_objects={reset_count}, restored_rigid_subprims={restored_rigid_count}"
            )
            officechair_prim = stage.GetPrimAtPath("/World/envs/env_0/Room/model_officechair_3")
            if officechair_prim.IsValid() and officechair_prim.IsActive():
                officechair_xf = UsdGeom.Xformable(officechair_prim)
                officechair_translate = None
                for op in officechair_xf.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        officechair_translate = op.Get()
                        break
                if officechair_translate is None:
                    officechair_translate = (0.0, 0.0, 0.0)
                print(
                    "[livingroom_post_reset] officechair local_pos="
                    f"({float(officechair_translate[0]):.4f}, {float(officechair_translate[1]):.4f}, {float(officechair_translate[2]):.4f})"
                )
            sofa_states = []
            for sofa_path in (
                "/World/envs/env_0/Room/model_smallsofa",
                "/World/envs/env_0/Room/model_smallsofa_01",
            ):
                sofa_prim = stage.GetPrimAtPath(sofa_path)
                sofa_states.append(f"{sofa_path.split('/')[-1]}={'active' if sofa_prim.IsValid() and sofa_prim.IsActive() else 'inactive'}")
            print(f"[livingroom_post_reset] sofa_states: {', '.join(sofa_states)}")

            xform_cache = UsdGeom.XformCache()

            def _stage_world_pos(prim_path: str):
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid() or (not prim.IsActive()):
                    return None
                world_tf = xform_cache.GetLocalToWorldTransform(prim)
                world_t = world_tf.ExtractTranslation()
                return [float(world_t[0]), float(world_t[1]), float(world_t[2])]

            try:
                table_drink = env.scene["table_drink"]
                table_pos = table_drink.data.root_state_w[0, :3].tolist()
                print(f"[livingroom_post_reset] table_drink root_pos={table_pos}")
            except Exception as exc:
                table_stage_pos = _stage_world_pos("/World/envs/env_0/TableDrink")
                print(
                    "[livingroom_post_reset] table_drink root_pos unavailable: "
                    f"{exc}; stage_world_pos={table_stage_pos}"
                )
            try:
                obstacle_1 = env.scene["floor_obstacle_drink_1"]
                obstacle_1_pos = obstacle_1.data.root_state_w[0, :3].tolist()
                print(f"[livingroom_post_reset] floor_obstacle_drink_1 root_pos={obstacle_1_pos}")
            except Exception as exc:
                obstacle_1_stage_pos = _stage_world_pos("/World/envs/env_0/FloorObstacleDrink1")
                print(
                    "[livingroom_post_reset] floor_obstacle_drink_1 root_pos unavailable: "
                    f"{exc}; stage_world_pos={obstacle_1_stage_pos}"
                )
        except Exception as exc:
            print(f"[livingroom_post_reset] verification failed: {exc}")

        if os.environ.get("LIVINGROOM_RUNTIME_PATCH", "0") == "1":
            try:
                import omni.usd
                from pxr import UsdGeom

                stage = omni.usd.get_context().get_stage()
                xform_cache = UsdGeom.XformCache()
                target_paths = (
                    "/World/envs/env_0/Room/model_teatable",
                    "/World/envs/env_0/Room/model_teatable/E_body_179",
                    "/World/envs/env_0/Room/model_table_1",
                    "/World/envs/env_0/Room/model_table_1/E_body_1",
                )
                print("[scene_probe] begin teatable/table_1 world pose dump")
                for prim_path in target_paths:
                    prim = stage.GetPrimAtPath(prim_path)
                    if not prim.IsValid() or (not prim.IsActive()):
                        continue
                    world_m = xform_cache.GetLocalToWorldTransform(prim)
                    world_t = world_m.ExtractTranslation()
                    print(
                        f"[scene_probe] {prim_path} "
                        f"world_pos=({float(world_t[0]):.4f}, {float(world_t[1]):.4f}, {float(world_t[2]):.4f})"
                    )
                print("[scene_probe] end teatable/table_1 world pose dump")
            except Exception as exc:
                print(f"[scene_probe] failed: {exc}")
