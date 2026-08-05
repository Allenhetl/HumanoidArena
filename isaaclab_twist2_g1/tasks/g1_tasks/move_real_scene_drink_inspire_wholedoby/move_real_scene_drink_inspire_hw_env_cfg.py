import os

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_scene.base_scene_real_scene_lab import (
    RealSceneLabSceneCfg,
    _resolve_robot_init_pos,
    _resolve_robot_init_rot,
)

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "")

# ipark floor: floor_z = -1.124 (robot pelvis spawns at -0.335 = floor + 0.79).
IPARK_FLOOR_Z = -1.124
# SAM3D-reconstructed desk asset (desk0.usd): z-up bbox z in [-0.5, +0.497],
# i.e. local origin = table mid-height (leg base at -0.5, tabletop at +0.497).
DESK_USD = os.path.join(
    PROJECT_ROOT, "assets", "objects", "desk_rec", "desk0.usd"
)
DESK_HALF_HEIGHT = 0.5
DESK_TOPOFFSET = 0.497  # tabletop local z
TABLE_CENTER_POS = [-3.239, -3.2, IPARK_FLOOR_Z + DESK_HALF_HEIGHT]
TABLE_TOP_Z = IPARK_FLOOR_Z + DESK_HALF_HEIGHT + DESK_TOPOFFSET
DRINK_BODY_POS = [-3.239, -3.2, TABLE_TOP_Z]  # bottle bottom on tabletop

DRINK_BODY_USD = os.path.join(
    PROJECT_ROOT, "assets", "objects", "drink101", "drink101_body.usd"
)
DRINK_CAP_USD = os.path.join(
    PROJECT_ROOT, "assets", "objects", "drink101", "drink101_cap.usd"
)
DRINK_CAP_OFFSET_Z = 0.2649


@configclass
class RealSceneDrinkTaskSceneCfg(RealSceneLabSceneCfg):
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_wholebody(
        init_pos=_resolve_robot_init_pos(),
        init_rot=_resolve_robot_init_rot(),
    )

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=10,
        track_air_time=True,
        debug_vis=False,
    )

    front_camera = CameraPresets.g1_front_camera()

    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/DrinkTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=TABLE_CENTER_POS,
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=DESK_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                retain_accelerations=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
        ),
    )

    drink_body = RigidObjectCfg(
        prim_path="/World/envs/env_.*/DrinkBody",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=DRINK_BODY_POS,
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=DRINK_BODY_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            activate_contact_sensors=False,
        ),
    )

    drink_cap = RigidObjectCfg(
        prim_path="/World/envs/env_.*/DrinkCap",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[DRINK_BODY_POS[0], DRINK_BODY_POS[1], DRINK_BODY_POS[2] + DRINK_CAP_OFFSET_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(
            usd_path=DRINK_CAP_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                retain_accelerations=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,
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
        robot_inspire_state = ObsTerm(func=mdp.get_robot_inspire_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    fall = DoneTerm(func=mdp.fall_detected)


@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward, weight=1.0)


@configclass
class EventCfg:
    pass


@configclass
class MoveRealSceneDrinkG129InspireWholedobyEnvCfg(ManagerBasedRLEnvCfg):
    scene: RealSceneDrinkTaskSceneCfg = RealSceneDrinkTaskSceneCfg(
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
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        ref_dt = self.sim.dt
        self.scene.contact_forces.update_period = ref_dt
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physx.bounce_threshold_velocity = 0.01
        path = 1024 * 1024 * 4
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = path
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(func=lambda env: self._reset_object_self(env)),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(func=lambda env: self._reset_all_self(env)),
        )

    def initialize_task_scene(self, env, args_cli=None):
        pass

    def _reset_object_self(self, env):
        self._reset_drink_objects(env)

    def _reset_all_self(self, env):
        base_mdp.reset_scene_to_default(
            env,
            torch.arange(env.num_envs, device=env.device),
        )
        self._reset_drink_objects(env)

    def _reset_drink_objects(self, env):
        """Restore bottle + cap to their default root states with zero velocity.

        After a cap separation episode the cap may be far away / tumbling; an
        explicit zero-velocity restore keeps multi-episode runs stable.
        """
        try:
            env_ids = torch.arange(env.num_envs, device=env.device)
            for name in ("drink_body", "drink_cap"):
                asset = env.scene[name]
                default_root = asset.data.default_root_state.clone()
                default_root[:, 7:13] = 0.0  # zero linear + angular velocity
                asset.write_root_state_to_sim(default_root, env_ids=env_ids)
            env.scene.write_data_to_sim()
        except Exception as exc:
            print(f"[drink] object reset failed: {exc}")
