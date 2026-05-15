from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


THIS_DIR = Path(__file__).resolve().parent
TARGET_MODULE_NAME = (
    "isaaclab_twist2_g1.tasks.g1_tasks.move_open_door_g1_29dof_dex3_wholebody."
    "move_open_door_g1_29dof_dex3_hw_env_cfg"
)
TARGET_MODULE_PATH = (
    THIS_DIR
    / "tasks"
    / "g1_tasks"
    / "move_open_door_g1_29dof_dex3_wholebody"
    / "move_open_door_g1_29dof_dex3_hw_env_cfg.py"
)


def _register_module(monkeypatch, name: str, module: ModuleType) -> ModuleType:
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_open_door_env_cfg_module(monkeypatch):
    isaaclab = _register_module(monkeypatch, "isaaclab", ModuleType("isaaclab"))
    isaaclab.__path__ = []  # type: ignore[attr-defined]

    assets = _register_module(monkeypatch, "isaaclab.assets", ModuleType("isaaclab.assets"))

    class ArticulationCfg:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.__dict__.update(kwargs)

    assets.ArticulationCfg = ArticulationCfg
    isaaclab.assets = assets  # type: ignore[attr-defined]

    envs = _register_module(monkeypatch, "isaaclab.envs", ModuleType("isaaclab.envs"))

    class ManagerBasedRLEnvCfg:
        def __init__(self, *args, **kwargs):
            self.sim = SimpleNamespace(
                dt=0.0,
                render_interval=0,
                physx=SimpleNamespace(
                    bounce_threshold_velocity=0.0,
                    gpu_found_lost_aggregate_pairs_capacity=0,
                    gpu_total_aggregate_pairs_capacity=0,
                    friction_correlation_distance=0.0,
                ),
                physics_material=SimpleNamespace(
                    static_friction=0.0,
                    dynamic_friction=0.0,
                    friction_combine_mode="",
                    restitution_combine_mode="",
                ),
            )
            self.scene = getattr(self.__class__, "scene", None)

    envs.ManagerBasedRLEnvCfg = ManagerBasedRLEnvCfg
    isaaclab.envs = envs  # type: ignore[attr-defined]

    envs_mdp = _register_module(monkeypatch, "isaaclab.envs.mdp", ModuleType("isaaclab.envs.mdp"))

    def reset_scene_to_default(env, env_ids):
        return None

    envs_mdp.reset_scene_to_default = reset_scene_to_default

    managers = _register_module(monkeypatch, "isaaclab.managers", ModuleType("isaaclab.managers"))

    class _CfgBase:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.__dict__.update(kwargs)

    class ObservationGroupCfg(_CfgBase):
        enable_corruption = False
        concatenate_terms = False

    class ObservationTermCfg(_CfgBase):
        pass

    class RewardTermCfg(_CfgBase):
        pass

    class EventTermCfg(_CfgBase):
        pass

    managers.EventTermCfg = EventTermCfg
    managers.ObservationGroupCfg = ObservationGroupCfg
    managers.ObservationTermCfg = ObservationTermCfg
    managers.RewardTermCfg = RewardTermCfg

    sensors = _register_module(monkeypatch, "isaaclab.sensors", ModuleType("isaaclab.sensors"))

    class ContactSensorCfg(_CfgBase):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.update_period = kwargs.get("update_period", None)

    sensors.ContactSensorCfg = ContactSensorCfg

    utils = _register_module(monkeypatch, "isaaclab.utils", ModuleType("isaaclab.utils"))

    def configclass(cls):
        return cls

    utils.configclass = configclass

    tasks_pkg = _register_module(monkeypatch, "tasks", ModuleType("tasks"))
    tasks_pkg.__path__ = []  # type: ignore[attr-defined]

    common_config = _register_module(monkeypatch, "tasks.common_config", ModuleType("tasks.common_config"))

    class CameraPresets:
        @staticmethod
        def g1_front_camera():
            return SimpleNamespace(name="front_camera")

        @staticmethod
        def g1_world_camera():
            return SimpleNamespace(name="world_camera")

    class G1RobotPresets:
        @staticmethod
        def g1_29dof_dex3_wholebody(*, init_pos, init_rot):
            return ArticulationCfg(init_pos=init_pos, init_rot=init_rot)

    common_config.CameraPresets = CameraPresets
    common_config.G1RobotPresets = G1RobotPresets

    common_event_pkg = _register_module(monkeypatch, "tasks.common_event", ModuleType("tasks.common_event"))
    common_event_pkg.__path__ = []  # type: ignore[attr-defined]
    common_event_manager = _register_module(
        monkeypatch,
        "tasks.common_event.event_manager",
        ModuleType("tasks.common_event.event_manager"),
    )

    class SimpleEvent:
        def __init__(self, func, params=None):
            self.func = func
            self.params = params or {}

        def trigger(self, env):
            return self.func(env, **self.params)

    class SimpleEventManager:
        def __init__(self):
            self._events = {}

        def register(self, name, event):
            self._events[name] = event

        def trigger(self, name, env):
            return self._events[name].trigger(env)

    common_event_manager.SimpleEvent = SimpleEvent
    common_event_manager.SimpleEventManager = SimpleEventManager

    common_scene_pkg = _register_module(monkeypatch, "tasks.common_scene", ModuleType("tasks.common_scene"))
    common_scene_pkg.__path__ = []  # type: ignore[attr-defined]
    base_scene_open_door = _register_module(
        monkeypatch,
        "tasks.common_scene.base_scene_open_door",
        ModuleType("tasks.common_scene.base_scene_open_door"),
    )

    class OpenDoorSceneCfg:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.__dict__.update(kwargs)

    base_scene_open_door.OpenDoorSceneCfg = OpenDoorSceneCfg

    common_env_objects = _register_module(monkeypatch, "common_env_objects", ModuleType("common_env_objects"))

    def apply_deterministic_object_resets(env_cfg, env, *, selected_record_names=None):
        return []

    common_env_objects.apply_deterministic_object_resets = apply_deterministic_object_resets

    root_pkg = _register_module(monkeypatch, "isaaclab_twist2_g1", ModuleType("isaaclab_twist2_g1"))
    root_pkg.__path__ = [str(THIS_DIR)]  # type: ignore[attr-defined]
    tasks_root_pkg = _register_module(
        monkeypatch,
        "isaaclab_twist2_g1.tasks",
        ModuleType("isaaclab_twist2_g1.tasks"),
    )
    tasks_root_pkg.__path__ = [str(THIS_DIR / "tasks")]  # type: ignore[attr-defined]
    g1_tasks_pkg = _register_module(
        monkeypatch,
        "isaaclab_twist2_g1.tasks.g1_tasks",
        ModuleType("isaaclab_twist2_g1.tasks.g1_tasks"),
    )
    g1_tasks_pkg.__path__ = [str(THIS_DIR / "tasks" / "g1_tasks")]  # type: ignore[attr-defined]
    open_door_pkg = _register_module(
        monkeypatch,
        "isaaclab_twist2_g1.tasks.g1_tasks.move_open_door_g1_29dof_dex3_wholebody",
        ModuleType("isaaclab_twist2_g1.tasks.g1_tasks.move_open_door_g1_29dof_dex3_wholebody"),
    )
    open_door_pkg.__path__ = [str(TARGET_MODULE_PATH.parent)]  # type: ignore[attr-defined]

    mdp_module = _register_module(
        monkeypatch,
        "isaaclab_twist2_g1.tasks.g1_tasks.move_open_door_g1_29dof_dex3_wholebody.mdp",
        ModuleType("isaaclab_twist2_g1.tasks.g1_tasks.move_open_door_g1_29dof_dex3_wholebody.mdp"),
    )

    class JointPositionActionCfg(_CfgBase):
        pass

    mdp_module.JointPositionActionCfg = JointPositionActionCfg
    mdp_module.get_robot_boy_joint_states = lambda *args, **kwargs: None
    mdp_module.get_robot_dex3_joint_states = lambda *args, **kwargs: None
    mdp_module.get_camera_image = lambda *args, **kwargs: None
    mdp_module.compute_reward_open_door = lambda *args, **kwargs: None

    spec = importlib.util.spec_from_file_location(TARGET_MODULE_NAME, TARGET_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, TARGET_MODULE_NAME, module)
    spec.loader.exec_module(module)
    return module


def test_open_door_event_cfg_is_empty_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPEN_DOOR_EVENT_SMOKE", raising=False)
    monkeypatch.delenv("OPEN_DOOR_LATCH_ENABLE", raising=False)
    monkeypatch.delenv("OPEN_DOOR_LATCH_DISABLE", raising=False)
    module = _load_open_door_env_cfg_module(monkeypatch)

    assert "open_door_event_smoke_startup" not in module.EventCfg.__dict__
    assert "open_door_event_smoke_reset" not in module.EventCfg.__dict__
    assert "open_door_event_smoke_interval" not in module.EventCfg.__dict__
    assert "open_door_latch_startup" not in module.EventCfg.__dict__
    assert "open_door_latch_reset" not in module.EventCfg.__dict__
    assert "open_door_latch_poll" not in module.EventCfg.__dict__


def test_open_door_event_smoke_registers_startup_only(monkeypatch) -> None:
    monkeypatch.setenv("OPEN_DOOR_EVENT_SMOKE", "1")
    monkeypatch.setenv("OPEN_DOOR_EVENT_SMOKE_MODE", "startup")
    module = _load_open_door_env_cfg_module(monkeypatch)

    assert module.EventCfg.open_door_event_smoke_startup.mode == "startup"
    assert module.EventCfg.open_door_event_smoke_startup.func is module._open_door_event_smoke
    assert "open_door_event_smoke_reset" not in module.EventCfg.__dict__
    assert "open_door_event_smoke_interval" not in module.EventCfg.__dict__


def test_open_door_latch_event_cfg_registers_startup_reset_and_interval(monkeypatch) -> None:
    monkeypatch.delenv("OPEN_DOOR_EVENT_SMOKE", raising=False)
    monkeypatch.setenv("OPEN_DOOR_LATCH_ENABLE", "1")
    module = _load_open_door_env_cfg_module(monkeypatch)

    assert module.EventCfg.open_door_latch_startup.mode == "startup"
    assert module.EventCfg.open_door_latch_reset.mode == "reset"
    assert module.EventCfg.open_door_latch_poll.mode == "interval"
    assert module.EventCfg.open_door_latch_poll.interval_range_s == (0.02, 0.02)


def test_open_door_latch_threshold_uses_signed_direction(monkeypatch) -> None:
    module = _load_open_door_env_cfg_module(monkeypatch)
    cfg = module.MoveOpenDoorG129Dex3WholebodyEnvCfg()
    cfg.__post_init__()

    cfg._open_door_handle_unlock_angle_deg = -20.0
    assert cfg._open_door_latch_unlock_threshold_met(-21.0)
    assert not cfg._open_door_latch_unlock_threshold_met(-19.0)

    cfg._open_door_handle_unlock_angle_deg = 20.0
    assert cfg._open_door_latch_unlock_threshold_met(21.0)
    assert not cfg._open_door_latch_unlock_threshold_met(19.0)


def test_sync_reward_after_physics_step_runs_open_door_post_physics_hook() -> None:
    from tools.get_reward import sync_reward_after_physics_step

    calls = []
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            apply_open_door_latch_interval=lambda env_arg, reason: calls.append((env_arg, reason)),
        ),
    )

    sync_reward_after_physics_step(env)

    assert calls == [(env, "manual_post_physics")]


def test_open_door_initialize_task_scene_sets_replay_flag(monkeypatch) -> None:
    module = _load_open_door_env_cfg_module(monkeypatch)
    cfg = module.MoveOpenDoorG129Dex3WholebodyEnvCfg()
    cfg.__post_init__()

    calls = []
    monkeypatch.setattr(cfg, "_disable_overlapping_room_gate_collisions", lambda: calls.append("gate"))
    monkeypatch.setattr(cfg, "_configure_door_joint_physics", lambda env: calls.append(("physics", env)))

    env = SimpleNamespace(name="env")
    args_cli = SimpleNamespace(replay_file="/tmp/open-door.npz")

    cfg.initialize_task_scene(env, args_cli=args_cli)

    assert cfg._replay_initial_env_state_active is True
    assert calls == ["gate", ("physics", env)]


def test_open_door_reset_object_self_uses_standard_deterministic_reset(monkeypatch) -> None:
    module = _load_open_door_env_cfg_module(monkeypatch)
    cfg = module.MoveOpenDoorG129Dex3WholebodyEnvCfg()
    cfg.__post_init__()
    env = SimpleNamespace(num_envs=1, device="cpu")

    calls = []

    def fake_apply(env_cfg, env_arg, *, selected_record_names=None):
        calls.append((env_cfg, env_arg, selected_record_names))
        return ["door->door"]

    monkeypatch.setattr(module, "apply_deterministic_object_resets", fake_apply)

    cfg._reset_object_self(env)

    assert len(calls) == 1
    assert calls[0][0] is cfg
    assert calls[0][1] is env
    assert calls[0][2] == {"door"}


def test_open_door_reset_all_self_resets_scene_then_randomizes(monkeypatch) -> None:
    module = _load_open_door_env_cfg_module(monkeypatch)
    cfg = module.MoveOpenDoorG129Dex3WholebodyEnvCfg()
    cfg.__post_init__()
    env = SimpleNamespace(num_envs=3, device="cpu")

    calls = []

    def fake_reset_scene_to_default(env_arg, env_ids):
        calls.append(("base_reset", env_arg, env_ids.clone()))

    def fake_apply(env_cfg, env_arg, *, selected_record_names=None):
        calls.append(("object_reset", env_cfg, env_arg, selected_record_names))
        return ["door->door"]

    monkeypatch.setattr(module.base_mdp, "reset_scene_to_default", fake_reset_scene_to_default)
    monkeypatch.setattr(module, "apply_deterministic_object_resets", fake_apply)

    cfg._reset_all_self(env)

    assert calls[0][0] == "base_reset"
    assert calls[0][1] is env
    torch.testing.assert_close(calls[0][2], torch.arange(env.num_envs, device=env.device))
    assert calls[1] == ("object_reset", cfg, env, {"door"})
