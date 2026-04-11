from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp
from tasks.common_config import CameraPresets, G1RobotPresets
from tasks.common_scene.base_scene_open_door import OpenDoorSceneCfg

ROBOT_INIT_POS = (-1.6, 0.2, 0.8)
ROBOT_INIT_ROT = (0.70711, 0.0, 0.0, 0.70711)
DOOR_LEAF_JOINT_STATIC_FRICTION = 0.0
DOOR_LEAF_JOINT_DYNAMIC_FRICTION = 0.0
DOOR_LEAF_JOINT_VISCOUS_FRICTION = 0.0
DOOR_HANDLE_JOINT_STATIC_FRICTION = 2.0
DOOR_HANDLE_JOINT_DYNAMIC_FRICTION = 1.5
DOOR_HANDLE_JOINT_VISCOUS_FRICTION = 0.5


@configclass
class OpenDoorTerrainSceneCfg(OpenDoorSceneCfg):
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
    reward = RewTerm(func=mdp.compute_reward_open_door, weight=1.0)


@configclass
class EventCfg:
    pass


@configclass
class MoveOpenDoorG129Dex3WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    scene: OpenDoorTerrainSceneCfg = OpenDoorTerrainSceneCfg(
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
        # Allow YAML override via tasks/common_env_config/*.yaml.
        self.object_reset_seed_source = "env_seed"

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

    def initialize_task_scene(self, env, args_cli=None):
        self._disable_overlapping_room_gate_collisions()
        self._configure_door_joint_physics(env)

    def _disable_overlapping_room_gate_collisions(self):
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            room_root = stage.GetPrimAtPath("/World/envs/env_0/Room")
            if not room_root or not room_root.IsValid():
                print("[open_door] room root not found; skipping gate collision override")
                return

            gate_roots = []
            stack = [room_root]
            while stack:
                current = stack.pop()
                if current.GetName().lower() == "gate":
                    gate_roots.append(current)
                stack.extend(list(current.GetChildren()))

            if not gate_roots:
                print("[open_door] no gate subtree found under room root; skipping gate collision override")
                return

            disabled_count = 0
            gate_paths = [str(prim.GetPath()) for prim in gate_roots]
            for gate_root in gate_roots:
                stack = [gate_root]
                while stack:
                    current = stack.pop()
                    collision_attr = current.GetAttribute("physics:collisionEnabled")
                    if collision_attr.IsValid() and collision_attr.Get() is not False:
                        collision_attr.Set(False)
                        disabled_count += 1
                    stack.extend(list(current.GetChildren()))

            print(
                "[open_door] disabled collision on "
                f"{disabled_count} prims under gate subtrees: {gate_paths}"
            )
        except Exception as exc:
            print(f"[open_door] failed to disable room gate collisions: {exc}")

    def _configure_door_joint_physics(self, env):
        try:
            import omni.usd
            from pxr import PhysxSchema, UsdPhysics

            stage = omni.usd.get_context().get_stage()
            door_asset = env.scene["door"]
            door_joint_ids, _ = door_asset.find_joints(["RevoluteJoint_door001"], preserve_order=True)
            handle_joint_ids, _ = door_asset.find_joints(["RevoluteJoint_handle001"], preserve_order=True)
            door_joint = stage.GetPrimAtPath(
                "/World/envs/env_0/Door/E_leaf_2/RevoluteJoint_door001"
            )
            handle_joint = stage.GetPrimAtPath("/World/envs/env_0/Door/RevoluteJoint_handle001")
            if not door_joint or not door_joint.IsValid():
                print("[open_door] door leaf joint not found; skipping drive override")
                return
            if not handle_joint or not handle_joint.IsValid():
                print("[open_door] handle joint not found; skipping handle friction override")

            angular_drive = UsdPhysics.DriveAPI.Get(door_joint, "angular")
            if not angular_drive:
                angular_drive = UsdPhysics.DriveAPI.Apply(door_joint, "angular")

            # The USD asset ships with a closing drive on the leaf joint
            # (stiffness=5, damping=20, max_force=5), which makes the door feel
            # artificially locked during teleoperation. Zero it out so the leaf
            # behaves as a passive hinge while the articulation uses a fixed base.
            angular_drive.GetTypeAttr().Set("force")
            angular_drive.GetStiffnessAttr().Set(0.0)
            angular_drive.GetDampingAttr().Set(0.0)
            angular_drive.GetMaxForceAttr().Set(0.0)
            angular_drive.GetTargetPositionAttr().Set(0.0)
            angular_drive.GetTargetVelocityAttr().Set(0.0)

            # Use passive joint physics instead of implicit actuators so the door
            # is driven purely by contact.
            door_asset.write_joint_stiffness_to_sim(0.0, joint_ids=door_joint_ids)
            door_asset.write_joint_damping_to_sim(0.0, joint_ids=door_joint_ids)
            door_asset.write_joint_friction_coefficient_to_sim(
                DOOR_LEAF_JOINT_STATIC_FRICTION,
                joint_dynamic_friction_coeff=DOOR_LEAF_JOINT_DYNAMIC_FRICTION,
                joint_viscous_friction_coeff=DOOR_LEAF_JOINT_VISCOUS_FRICTION,
                joint_ids=door_joint_ids,
            )

            door_asset.write_joint_stiffness_to_sim(0.0, joint_ids=handle_joint_ids)
            door_asset.write_joint_damping_to_sim(0.0, joint_ids=handle_joint_ids)
            door_asset.write_joint_friction_coefficient_to_sim(
                DOOR_HANDLE_JOINT_STATIC_FRICTION,
                joint_dynamic_friction_coeff=DOOR_HANDLE_JOINT_DYNAMIC_FRICTION,
                joint_viscous_friction_coeff=DOOR_HANDLE_JOINT_VISCOUS_FRICTION,
                joint_ids=handle_joint_ids,
            )

            leaf_joint_api = PhysxSchema.PhysxJointAPI.Apply(door_joint)
            handle_friction_value = None
            if handle_joint and handle_joint.IsValid():
                handle_joint_api = PhysxSchema.PhysxJointAPI.Apply(handle_joint)
                handle_friction_value = handle_joint_api.GetJointFrictionAttr().Get()

            leaf_stiffness = door_asset.data.joint_stiffness[0, door_joint_ids[0]].item()
            leaf_damping = door_asset.data.joint_damping[0, door_joint_ids[0]].item()
            leaf_static_friction = door_asset.data.joint_friction_coeff[0, door_joint_ids[0]].item()
            handle_stiffness = door_asset.data.joint_stiffness[0, handle_joint_ids[0]].item()
            handle_damping = door_asset.data.joint_damping[0, handle_joint_ids[0]].item()
            handle_static_friction = door_asset.data.joint_friction_coeff[0, handle_joint_ids[0]].item()

            joint_names = None
            try:
                joint_names = list(env.scene["door"].data.joint_names)
            except Exception:
                joint_names = None

            print(
                "[open_door] configured door joints: "
                f"joint_names={joint_names}, "
                f"leaf_drive=(stiffness={angular_drive.GetStiffnessAttr().Get()}, "
                f"damping={angular_drive.GetDampingAttr().Get()}, "
                f"max_force={angular_drive.GetMaxForceAttr().Get()}), "
                f"leaf_runtime=(stiffness={leaf_stiffness}, damping={leaf_damping}, "
                f"static_friction={leaf_static_friction}), "
                f"leaf_joint_friction_attr={leaf_joint_api.GetJointFrictionAttr().Get()}, "
                f"handle_runtime=(stiffness={handle_stiffness}, damping={handle_damping}, "
                f"static_friction={handle_static_friction}), "
                f"handle_joint_friction_attr={handle_friction_value}"
            )
        except Exception as exc:
            print(f"[open_door] failed to configure door joint physics: {exc}")
