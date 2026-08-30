from __future__ import annotations

import ast
import importlib.util
import io
import random
import sys
import types
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
RECOVERY_STATE_PATH = (
    THIS_DIR
    / "tasks"
    / "g1_tasks"
    / "move_pickplace_box_g1_29dof_dex3_wholedoby"
    / "mdp"
    / "recovery_state.py"
)
PP_BOX_CFG_PATH = (
    THIS_DIR
    / "tasks"
    / "g1_tasks"
    / "move_pickplace_box_g1_29dof_dex3_wholedoby"
    / "move_pickplace_box_g1_29dof_dex3_hw_env_cfg.py"
)
SIM_MAIN_PATH = THIS_DIR / "sim_main.py"
SIM_EVAL_VLA_PATH = THIS_DIR / "script" / "eval_scripts" / "sonic" / "sim_eval_vla.py"


def _load_recovery_state_module():
    spec = importlib.util.spec_from_file_location(
        "pp_box_recovery_state", RECOVERY_STATE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_recovery_value_digest_is_stable_for_nested_tensor_array_mapping_tuple_and_rng() -> (
    None
):
    recovery_state = _load_recovery_state_module()
    value = {
        "tensor": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "array": np.array([[3, 4]], dtype=np.int32),
        "tuple": ("phase", 7, False),
        "rng": {
            "python": random.Random(11).getstate(),
            "numpy": np.random.RandomState(12).get_state(),
            "torch": torch.Generator().manual_seed(13).get_state(),
        },
    }
    reordered = {
        "rng": recovery_state.clone_recovery_value(value["rng"]),
        "tuple": tuple(value["tuple"]),
        "array": value["array"].copy(),
        "tensor": value["tensor"].clone(),
    }

    first = recovery_state.recovery_value_digest(value)
    second = recovery_state.recovery_value_digest(reordered)

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")
    mutated = recovery_state.clone_recovery_value(value)
    mutated["tensor"][0, 1] = 2.5
    assert recovery_state.recovery_value_digest(mutated) != first


def test_tensor_digest_is_device_and_cpu_map_location_independent() -> None:
    recovery_state = _load_recovery_state_module()
    source = torch.tensor([[1.25, -2.5]], dtype=torch.float32)
    buffer = io.BytesIO()
    torch.save(source, buffer)
    buffer.seek(0)
    loaded_cpu = torch.load(buffer, map_location="cpu", weights_only=True)

    expected = recovery_state.recovery_value_digest(source)

    assert recovery_state.recovery_value_digest(loaded_cpu) == expected
    if torch.cuda.is_available():
        assert recovery_state.recovery_value_digest(source.cuda()) == expected


def test_recovery_state_digest_binds_every_snapshot_payload_without_object_identity() -> (
    None
):
    recovery_state = _load_recovery_state_module()
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(
        env,
        required_capabilities={"task_state", "task_local_rng", "wrapper_rng"},
    )
    cloned = replace(
        snapshot,
        scene_state={
            key: recovery_state.clone_recovery_value(snapshot.scene_state[key])
            for key in reversed(tuple(snapshot.scene_state))
        },
        task_counters=recovery_state.clone_recovery_value(snapshot.task_counters),
        task_state=recovery_state.clone_recovery_value(snapshot.task_state),
        rng_state=recovery_state.clone_recovery_value(snapshot.rng_state),
        runtime_state=recovery_state.clone_recovery_value(snapshot.runtime_state),
    )

    digest = recovery_state.recovery_state_digest(snapshot)

    assert recovery_state.recovery_state_digest(cloned) == digest
    mutated_task = recovery_state.clone_recovery_value(snapshot.task_state)
    mutated_task["debug_score"] += 1
    assert (
        recovery_state.recovery_state_digest(replace(snapshot, task_state=mutated_task))
        != digest
    )
    mutated_rng = replace(
        snapshot.rng_state,
        torch_cpu=torch.Generator().manual_seed(999).get_state(),
    )
    assert (
        recovery_state.recovery_state_digest(replace(snapshot, rng_state=mutated_rng))
        != digest
    )


class _Scene:
    def __init__(self, env: _FakeEnv) -> None:
        self._env = env

    def get_state(self, *, is_relative: bool | None = None) -> dict[str, object]:
        self._env.api_events.append(("get_state", is_relative))
        return {
            "robot": {"joint_pos": self._env.joint_pos},
            "box": {"root_state": self._env.box_root_state},
        }


class _FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.recovery_process_global_rng_exclusive = True
        self.cfg = SimpleNamespace(
            env_name="Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
            recovery_task_identity="Isaac-Move-PickPlace-Box-G129-Dex3-Wholedoby",
        )
        self.joint_pos = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        self.box_root_state = np.arange(13, dtype=np.float32).reshape(1, 13)
        self.episode_length_buf = torch.tensor([7], dtype=torch.int64)
        self.__dict__["common_step_counter"] = 11
        self.reset_buf = torch.tensor([False])
        self.episode_count = 3
        self.reset_count = torch.tensor([2], dtype=torch.int64)
        self.recovery_task_state = {
            "task_success_enabled": True,
            "debug_score": 5,
            "hands_materialized": torch.tensor([[True, False]]),
        }
        self.task_generator = torch.Generator().manual_seed(101)
        self.wrapper_rng = np.random.default_rng(202)
        self.scene = _Scene(self)
        self.restore_events: list[str] = []
        self.api_events: list[tuple[object, ...]] = []
        self.last_reset_state: dict[str, object] | None = None
        self.last_reset_env_ids: torch.Tensor | None = None
        self.preflight_failure: Exception | None = None
        self.restore_failures_remaining = 0
        self.corrupt_restores_remaining = 0

    def reset_to(
        self,
        state: dict[str, object],
        env_ids: torch.Tensor,
        *,
        is_relative: bool,
    ) -> None:
        self.api_events.append(
            (
                "reset_to",
                None if env_ids is None else env_ids.detach().cpu().clone(),
                is_relative,
            )
        )
        self.last_reset_state = state
        self.last_reset_env_ids = env_ids
        self.restore_events.append("scene")
        self.joint_pos = state["robot"]["joint_pos"].clone()
        self.box_root_state = state["box"]["root_state"].copy()
        # Real reset paths may consume RNG and mutate counters before callers restore them.
        random.random()
        np.random.random()
        torch.rand(1)
        torch.rand(1, generator=self.task_generator)
        wrapper_rng = getattr(self, "wrapper_rng", None)
        if wrapper_rng is not None:
            wrapper_rng.random()
        np_random = getattr(self, "np_random", None)
        if np_random is not None:
            np_random.random()
        self.episode_length_buf.add_(100)

    def capture_recovery_task_state(self) -> dict[str, object]:
        return self.recovery_task_state

    def preflight_restore_recovery_task_state(self, state: dict[str, object]) -> None:
        if self.preflight_failure is not None:
            raise self.preflight_failure
        assert set(state) == {
            "task_success_enabled",
            "debug_score",
            "hands_materialized",
        }

    def restore_recovery_task_state(self, state: dict[str, object]) -> None:
        self.restore_events.append("task")
        self.recovery_task_state = state
        if self.restore_failures_remaining:
            self.restore_failures_remaining -= 1
            raise RuntimeError("injected task-state restore failure")
        if self.corrupt_restores_remaining:
            self.corrupt_restores_remaining -= 1
            self.recovery_task_state["debug_score"] += 1

    def deterministic_step(self) -> dict[str, Any]:
        python_draw = random.random()
        numpy_draw = float(np.random.random())
        torch_draw = torch.rand(2)
        task_draw = torch.rand(2, generator=self.task_generator)
        wrapper_draw = self.wrapper_rng.random(2)

        self.joint_pos.add_(0.25 + python_draw * 0.01)
        self.box_root_state[:, :3] += numpy_draw * 0.01
        self.episode_length_buf.add_(1)
        self.common_step_counter += 1
        self.episode_count += 1
        self.reset_count.add_(self.reset_buf.to(torch.int64))
        self.recovery_task_state["debug_score"] += 1
        hands = self.recovery_task_state["hands_materialized"]
        observation = torch.tensor(
            [
                float(self.joint_pos.sum()),
                float(self.box_root_state[:, :3].sum()),
                float(self.episode_length_buf.item()),
                float(self.common_step_counter),
                float(self.episode_count),
                float(self.reset_count.item()),
                float(self.recovery_task_state["debug_score"]),
                float(hands.to(torch.float32).sum()),
                python_draw,
                numpy_draw,
                float(torch_draw.sum()),
                float(task_draw.sum()),
                float(wrapper_draw.sum()),
            ],
            dtype=torch.float64,
        )
        reward = float(observation.sum())
        terminal = (
            "success"
            if self.recovery_task_state["task_success_enabled"]
            and bool(hands[0, 0])
            and not bool(self.reset_buf[0])
            else "running"
        )
        return {
            "scene": {
                "joint_pos": self.joint_pos.clone(),
                "box_root_state": self.box_root_state.copy(),
            },
            "counters": {
                "episode_length_buf": self.episode_length_buf.clone(),
                "common_step_counter": self.common_step_counter,
                "reset_buf": self.reset_buf.clone(),
                "episode_count": self.episode_count,
                "reset_count": self.reset_count.clone(),
            },
            "task_state": {
                "task_success_enabled": self.recovery_task_state[
                    "task_success_enabled"
                ],
                "debug_score": self.recovery_task_state["debug_score"],
                "hands_materialized": hands.clone(),
            },
            "rng": {
                "python": python_draw,
                "numpy": numpy_draw,
                "torch": torch_draw,
                "task": task_draw,
                "wrapper": wrapper_draw,
            },
            "observation": observation,
            "reward": reward,
            "terminal": terminal,
        }


class _CacheEnv(_FakeEnv):
    def capture_physics_solver_cache(self) -> dict[str, int]:
        return {"solver_epoch": 4}

    def restore_physics_solver_cache(self, state: dict[str, int]) -> None:
        self.restore_events.append(f"solver:{state['solver_epoch']}")

    def preflight_restore_physics_solver_cache(self, state: dict[str, int]) -> None:
        assert set(state) == {"solver_epoch"}


class _BothCacheEnv(_CacheEnv):
    def capture_contact_cache(self) -> dict[str, int]:
        return {"contact_epoch": 6}

    def restore_contact_cache(self, state: dict[str, int]) -> None:
        self.restore_events.append(f"contact:{state['contact_epoch']}")

    def preflight_restore_contact_cache(self, state: dict[str, int]) -> None:
        assert set(state) == {"contact_epoch"}


class _ControlParticipant:
    def __init__(self, capability: str, value: int) -> None:
        self.capability = capability
        self.value = value
        self.restore_failures_remaining = 0

    def _capture(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "capability": self.capability,
            "value": self.value,
        }

    def _preflight(self, state: dict[str, object]) -> None:
        if set(state) != {"schema_version", "capability", "value"}:
            raise ValueError("participant schema mismatch")
        if state["schema_version"] != 1 or state["capability"] != self.capability:
            raise ValueError("participant identity mismatch")
        if type(state["value"]) is not int:
            raise ValueError("participant value mismatch")

    def _restore(self, state: dict[str, object]) -> None:
        self.value = int(state["value"])
        if self.restore_failures_remaining:
            self.restore_failures_remaining -= 1
            raise RuntimeError(f"injected {self.capability} restore failure")


class _ProviderParticipant(_ControlParticipant):
    def __init__(self, value: int) -> None:
        super().__init__("action_provider_state", value)

    def capture_recovery_provider_state(self):
        return self._capture()

    def preflight_restore_recovery_provider_state(self, state):
        self._preflight(state)

    def restore_recovery_provider_state(self, state):
        self._restore(state)


class _ControllerParticipant(_ControlParticipant):
    def __init__(self, provider: _ProviderParticipant, value: int) -> None:
        super().__init__("controller_state", value)
        self.action_provider = provider

    def capture_recovery_controller_state(self):
        return self._capture()

    def preflight_restore_recovery_controller_state(self, state):
        self._preflight(state)

    def restore_recovery_controller_state(self, state):
        self._restore(state)


class _ActionManagerParticipant(_ControlParticipant):
    def __init__(self, value: int) -> None:
        super().__init__("action_manager_state", value)

    def capture_recovery_action_manager_state(self):
        return self._capture()

    def preflight_restore_recovery_action_manager_state(self, state):
        self._preflight(state)

    def restore_recovery_action_manager_state(self, state):
        self._restore(state)


class _ExactControlEnv(_BothCacheEnv):
    def __init__(self) -> None:
        super().__init__()
        self.action_provider = _ProviderParticipant(31)
        self.recovery_controller = _ControllerParticipant(self.action_provider, 32)
        self.action_manager = _ActionManagerParticipant(33)


class _CaptureOnlyTaskStateEnv(_FakeEnv):
    restore_recovery_task_state = None

    def __init__(self) -> None:
        super().__init__()
        del self.recovery_task_state

    def capture_recovery_task_state(self) -> dict[str, str]:
        return {"capture_only_marker": "present"}


class _DirectTaskStateEnv(_FakeEnv):
    capture_recovery_task_state = None
    preflight_restore_recovery_task_state = None
    restore_recovery_task_state = None


class _ReadOnlyCounterEnv(_FakeEnv):
    def __init__(self) -> None:
        self._common_step_counter = 11
        super().__init__()

    @property
    def common_step_counter(self) -> int:
        return self._common_step_counter


class _BadResetSignatureEnv(_FakeEnv):
    def reset_to(self, state: dict[str, object]) -> None:
        self.restore_events.append("bad-scene")


class _ProductionRobot:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            joint_pos_target=torch.tensor([[16.0, 17.0]]),
            joint_vel_target=torch.tensor([[18.0, 19.0]]),
            joint_effort_target=torch.tensor([[20.0, 21.0]]),
        )

    def set_joint_position_target(self, target, *, env_ids=None) -> None:
        self.data.joint_pos_target[env_ids] = target

    def set_joint_velocity_target(self, target, *, env_ids=None) -> None:
        self.data.joint_vel_target[env_ids] = target

    def set_joint_effort_target(self, target, *, env_ids=None) -> None:
        self.data.joint_effort_target[env_ids] = target


class _ProductionScene(_Scene):
    def __init__(self, env: _ProductionEnv) -> None:
        super().__init__(env)
        self.robot = _ProductionRobot()

    def __getitem__(self, name: str):
        if name != "robot":
            raise KeyError(name)
        return self.robot


class JointPositionAction:
    action_dim = 2

    def __init__(self) -> None:
        self._raw_actions = torch.tensor([[5.0, 6.0]])
        self._processed_actions = torch.tensor([[7.0, 8.0]])


class _ProductionEnv:
    """Pure-Python mirror of PP-box buffers reset by Isaac Lab v2.2.1 reset_to."""

    def __init__(self, task_identity: str) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.joint_pos = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        self.box_root_state = np.arange(13, dtype=np.float32).reshape(1, 13)
        self.episode_length_buf = torch.tensor([7], dtype=torch.int64)
        self.common_step_counter = 11
        self.reset_buf = torch.tensor([False])
        self.api_events: list[tuple[object, ...]] = []
        self.scene = _ProductionScene(self)
        self.action_manager = SimpleNamespace(
            _term_names=["joint_pos"],
            _terms={"joint_pos": JointPositionAction()},
            _action=torch.tensor([[1.0, 2.0]]),
            _prev_action=torch.tensor([[3.0, 4.0]]),
        )
        self.reward_manager = SimpleNamespace(
            _term_names=["reward"],
            _class_term_cfgs=[],
            _episode_sums={"reward": torch.tensor([9.0])},
            _reward_buf=torch.tensor([10.0]),
            _step_reward=torch.tensor([[11.0]]),
        )
        self.termination_manager = SimpleNamespace(
            _term_names=[],
            _class_term_cfgs=[],
            _term_dones={},
            _truncated_buf=torch.tensor([False]),
            _terminated_buf=torch.tensor([True]),
        )
        observation = {
            "policy": {
                "robot_joint_state": torch.tensor([[12.0, 13.0]]),
                "robot_gipper_state": torch.tensor([[14.0, 15.0]]),
                "camera_image": torch.tensor([[[[16.0, 17.0, 18.0]]]]),
            }
        }
        self.observation_manager = SimpleNamespace(
            _group_obs_term_names={
                "policy": [
                    "robot_joint_state",
                    "robot_gipper_state",
                    "camera_image",
                ]
            },
            _group_obs_term_history_buffer={"policy": {}},
            _group_obs_class_term_cfgs={"policy": []},
            _group_obs_class_instances=[],
            _obs_buffer=observation,
        )
        for name in (
            "command_manager",
            "curriculum_manager",
            "event_manager",
            "recorder_manager",
        ):
            setattr(self, name, SimpleNamespace(active_terms=[]))
        self._sim_step_counter = 14
        self.extras = {"log": {"episode": torch.tensor([15.0])}}
        self.reward_buf = self.reward_manager._reward_buf
        self.reset_terminated = self.termination_manager._terminated_buf
        self.reset_time_outs = self.termination_manager._truncated_buf
        self.obs_buf = self.observation_manager._obs_buffer
        self.cfg = SimpleNamespace(
            env_name=task_identity,
            recovery_task_identity=task_identity,
            _episode_runtime_seed=22,
            _episode_object_seed_counter=23,
            _current_episode_object_seed=24,
            _current_episode_object_seed_source="env_seed",
            _replay_initial_env_state_active=True,
        )
        self.task_generator = torch.Generator().manual_seed(101)
        self.wrapper_rng = np.random.default_rng(202)
        self.reset_calls = 0

    def reset_to(self, state, env_ids, *, is_relative) -> None:
        assert torch.equal(env_ids, torch.tensor([0], device=self.device))
        assert is_relative is True
        self.reset_calls += 1
        self.joint_pos.copy_(state["robot"]["joint_pos"])
        np.copyto(self.box_root_state, state["box"]["root_state"])
        self.episode_length_buf.zero_()
        self.action_manager._action.zero_()
        self.action_manager._prev_action.zero_()
        term = self.action_manager._terms["joint_pos"]
        term._raw_actions.zero_()
        term._processed_actions.zero_()
        self.reward_manager._episode_sums["reward"].zero_()
        self.reward_manager._reward_buf.zero_()
        self.reward_manager._step_reward.zero_()
        self.termination_manager._truncated_buf.zero_()
        self.termination_manager._terminated_buf.zero_()
        self.observation_manager._obs_buffer = {
            "policy": {
                "robot_joint_state": torch.zeros(1, 2),
                "robot_gipper_state": torch.zeros(1, 2),
                "camera_image": torch.zeros(1, 1, 1, 3),
            }
        }
        self.obs_buf = self.observation_manager._obs_buffer
        self._sim_step_counter = 0
        self.extras = {"log": {}}
        self.scene.robot.data.joint_pos_target.zero_()
        self.scene.robot.data.joint_vel_target.zero_()
        self.scene.robot.data.joint_effort_target.zero_()
        self.cfg._episode_runtime_seed = -1
        self.cfg._episode_object_seed_counter = -1
        self.cfg._current_episode_object_seed = None
        self.cfg._current_episode_object_seed_source = "reset"
        self.cfg._replay_initial_env_state_active = False


class _ProductionExactEnv(_ProductionEnv):
    def __init__(self, task_identity: str) -> None:
        super().__init__(task_identity)
        self.physics_solver_epoch = 4
        self.contact_epoch = 6

    def capture_physics_solver_cache(self):
        return {"solver_epoch": self.physics_solver_epoch}

    def preflight_restore_physics_solver_cache(self, state):
        if set(state) != {"solver_epoch"} or type(state["solver_epoch"]) is not int:
            raise ValueError("solver cache schema mismatch")

    def restore_physics_solver_cache(self, state):
        self.physics_solver_epoch = state["solver_epoch"]

    def capture_contact_cache(self):
        return {"contact_epoch": self.contact_epoch}

    def preflight_restore_contact_cache(self, state):
        if set(state) != {"contact_epoch"} or type(state["contact_epoch"]) is not int:
            raise ValueError("contact cache schema mismatch")

    def restore_contact_cache(self, state):
        self.contact_epoch = state["contact_epoch"]


class _DelegatingGymWrapper:
    def __init__(self, unwrapped: _FakeEnv, nested_env: object) -> None:
        self.unwrapped = unwrapped
        self.env = nested_env
        self._runtime = unwrapped

    def __getattr__(self, name: str) -> object:
        return getattr(self._runtime, name)


class _EnvLink:
    def __init__(self) -> None:
        self.env: object | None = None


class _AbsoluteScene(_Scene):
    def get_state(self, *, is_relative: bool | None = None) -> dict[str, object]:
        self._env.api_events.append(("get_state", is_relative))
        offset = 0.0 if is_relative is True else 1000.0
        return {
            "robot": {"joint_pos": self._env.joint_pos + offset},
            "box": {"root_state": self._env.box_root_state + offset},
        }


def _global_rng_state() -> tuple[object, tuple[object, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_global_rng_state_equal(
    actual: tuple[object, tuple[object, ...], torch.Tensor],
    expected: tuple[object, tuple[object, ...], torch.Tensor],
) -> None:
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0]
    np.testing.assert_array_equal(actual[1][1], expected[1][1])
    assert actual[1][2:] == expected[1][2:]
    torch.testing.assert_close(actual[2], expected[2])


def _env_state(env: _FakeEnv) -> dict[str, Any]:
    task_state = getattr(env, "recovery_task_state", None)
    if task_state is not None:
        task_state = {
            "task_success_enabled": task_state["task_success_enabled"],
            "debug_score": task_state["debug_score"],
            "hands_materialized": task_state["hands_materialized"].clone(),
        }
    return {
        "joint_pos": env.joint_pos.clone(),
        "box_root_state": env.box_root_state.copy(),
        "episode_length_buf": env.episode_length_buf.clone(),
        "common_step_counter": env.common_step_counter,
        "reset_buf": env.reset_buf.clone(),
        "episode_count": env.episode_count,
        "reset_count": env.reset_count.clone(),
        "task_state": task_state,
        "task_rng": env.task_generator.get_state().clone(),
        "wrapper_rng": dict(env.wrapper_rng.bit_generator.state),
        "restore_events": list(env.restore_events),
    }


def _assert_env_state_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    torch.testing.assert_close(actual["joint_pos"], expected["joint_pos"])
    np.testing.assert_array_equal(actual["box_root_state"], expected["box_root_state"])
    torch.testing.assert_close(
        actual["episode_length_buf"], expected["episode_length_buf"]
    )
    assert actual["common_step_counter"] == expected["common_step_counter"]
    torch.testing.assert_close(actual["reset_buf"], expected["reset_buf"])
    assert actual["episode_count"] == expected["episode_count"]
    torch.testing.assert_close(actual["reset_count"], expected["reset_count"])
    if expected["task_state"] is None:
        assert actual["task_state"] is None
    else:
        assert (
            actual["task_state"]["task_success_enabled"]
            is (expected["task_state"]["task_success_enabled"])
        )
        assert (
            actual["task_state"]["debug_score"] == expected["task_state"]["debug_score"]
        )
        torch.testing.assert_close(
            actual["task_state"]["hands_materialized"],
            expected["task_state"]["hands_materialized"],
        )
    torch.testing.assert_close(actual["task_rng"], expected["task_rng"])
    assert actual["wrapper_rng"] == expected["wrapper_rng"]
    assert actual["restore_events"] == expected["restore_events"]


def _assert_transition_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    torch.testing.assert_close(
        actual["scene"]["joint_pos"], expected["scene"]["joint_pos"]
    )
    np.testing.assert_array_equal(
        actual["scene"]["box_root_state"], expected["scene"]["box_root_state"]
    )
    for name in ("episode_length_buf", "reset_buf", "reset_count"):
        torch.testing.assert_close(actual["counters"][name], expected["counters"][name])
    for name in ("common_step_counter", "episode_count"):
        assert actual["counters"][name] == expected["counters"][name]
    assert (
        actual["task_state"]["task_success_enabled"]
        is (expected["task_state"]["task_success_enabled"])
    )
    assert actual["task_state"]["debug_score"] == expected["task_state"]["debug_score"]
    torch.testing.assert_close(
        actual["task_state"]["hands_materialized"],
        expected["task_state"]["hands_materialized"],
    )
    assert actual["rng"]["python"] == expected["rng"]["python"]
    assert actual["rng"]["numpy"] == expected["rng"]["numpy"]
    torch.testing.assert_close(actual["rng"]["torch"], expected["rng"]["torch"])
    torch.testing.assert_close(actual["rng"]["task"], expected["rng"]["task"])
    np.testing.assert_array_equal(actual["rng"]["wrapper"], expected["rng"]["wrapper"])
    torch.testing.assert_close(actual["observation"], expected["observation"])
    assert actual["reward"] == expected["reward"]
    assert actual["terminal"] == expected["terminal"]


@pytest.fixture()
def recovery_state():
    return _load_recovery_state_module()


def _restore_bound(recovery_state, env, snapshot, **kwargs) -> None:
    recovery_state.restore_recovery_state(
        env,
        snapshot,
        snapshot_digest=recovery_state.recovery_state_digest(snapshot),
        **kwargs,
    )


def test_capture_and_restore_bind_relative_single_env_isaac_api(recovery_state) -> None:
    env = _FakeEnv()

    snapshot = recovery_state.capture_recovery_state(env)

    assert snapshot.scope.num_envs == 1
    assert snapshot.scope.env_ids == (0,)
    assert snapshot.scope.is_relative is True
    assert env.api_events == [
        ("get_state", True),
        ("get_state", True),
    ]
    env.api_events.clear()
    _restore_bound(recovery_state, env, snapshot)

    reset_events = [event for event in env.api_events if event[0] == "reset_to"]
    assert len(reset_events) == 1
    reset_event = reset_events[0]
    reset_index = env.api_events.index(reset_event)
    assert all(event == ("get_state", True) for event in env.api_events[:reset_index])
    assert all(
        event == ("get_state", True) for event in env.api_events[reset_index + 1 :]
    )
    torch.testing.assert_close(reset_event[1], torch.tensor([0]))
    assert reset_event[1].dtype == torch.long
    assert reset_event[1].device == env.device
    assert reset_event[2] is True
    assert env.last_reset_state is not snapshot.scene_state
    assert env.last_reset_state["robot"] is not snapshot.scene_state["robot"]
    assert env.last_reset_env_ids is not None
    assert env.last_reset_env_ids.device == env.device


def test_relative_scene_capture_restore_is_symmetric(recovery_state) -> None:
    env = _FakeEnv()
    env.scene = _AbsoluteScene(env)
    expected_joint_pos = env.joint_pos.clone()
    expected_box_root_state = env.box_root_state.copy()

    snapshot = recovery_state.capture_recovery_state(env)

    torch.testing.assert_close(
        snapshot.scene_state["robot"]["joint_pos"], expected_joint_pos
    )
    np.testing.assert_array_equal(
        snapshot.scene_state["box"]["root_state"],
        expected_box_root_state,
    )
    env.joint_pos.fill_(-100.0)
    env.box_root_state.fill(-200.0)
    _restore_bound(recovery_state, env, snapshot)

    torch.testing.assert_close(env.joint_pos, expected_joint_pos)
    np.testing.assert_array_equal(env.box_root_state, expected_box_root_state)
    reset_events = [event for event in env.api_events if event[0] == "reset_to"]
    assert len(reset_events) == 1
    assert reset_events[0][2] is True


def test_restore_rejects_digest_bound_scope_mismatch_before_environment_access(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    invalid_scope = replace(snapshot.scope, is_relative=False)
    invalid = replace(snapshot, scope=invalid_scope)
    env.api_events.clear()
    env_before = _env_state(env)

    with pytest.raises(recovery_state.RecoveryStateSchemaError, match="scope"):
        _restore_bound(recovery_state, env, invalid)

    assert env.api_events == []
    _assert_env_state_equal(_env_state(env), env_before)


def test_restore_rejects_actual_cfg_env_name_before_environment_access(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env.cfg.env_name = "Isaac-Another-Task"
    env.api_events.clear()
    env_before = _env_state(env)

    with pytest.raises(recovery_state.RecoveryStateSchemaError, match="cfg.env_name"):
        _restore_bound(recovery_state, env, snapshot)

    assert env.api_events == []
    _assert_env_state_equal(_env_state(env), env_before)


def test_restore_rejects_reset_to_without_explicit_env_ids_before_scene_access(
    recovery_state,
) -> None:
    source = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(source)
    env = _BadResetSignatureEnv()
    env.api_events.clear()
    env_before = _env_state(env)

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        _restore_bound(recovery_state, env, snapshot)

    assert "reset_to_signature" in exc_info.value.missing_capabilities
    assert env.api_events == []
    _assert_env_state_equal(_env_state(env), env_before)


@pytest.mark.parametrize(
    ("attribute", "value", "expected_missing"),
    [
        ("num_envs", None, "num_envs"),
        ("num_envs", 2, "num_envs"),
        ("device", None, "device"),
        ("device", "not-a-torch-device", "device"),
    ],
)
@pytest.mark.parametrize("operation", ["capture", "restore"])
def test_capture_and_restore_fail_closed_without_valid_single_env_selection(
    recovery_state,
    attribute: str,
    value: object,
    expected_missing: str,
    operation: str,
) -> None:
    source = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(source)
    env = _FakeEnv()
    if value is None:
        delattr(env, attribute)
    else:
        setattr(env, attribute, value)

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        if operation == "capture":
            recovery_state.capture_recovery_state(env)
        else:
            _restore_bound(recovery_state, env, snapshot)

    assert expected_missing in exc_info.value.missing_capabilities
    assert env.api_events == []
    assert env.restore_events == []


def test_unwrapped_np_random_is_bound_before_nested_or_forwarded_rng(
    recovery_state,
) -> None:
    runtime = _FakeEnv()
    runtime.np_random = np.random.default_rng(303)
    nested = _EnvLink()
    nested.np_random = np.random.default_rng(404)
    wrapper = _DelegatingGymWrapper(runtime, nested)
    snapshot = recovery_state.capture_recovery_state(
        wrapper,
        required_capabilities={"wrapper_rng"},
    )
    expected = runtime.np_random.random(3)
    runtime.np_random.random(7)
    runtime.wrapper_rng.random(5)
    nested.np_random.random(4)
    nested_state_before_restore = dict(nested.np_random.bit_generator.state)

    _restore_bound(recovery_state, wrapper, snapshot)

    np.testing.assert_array_equal(runtime.np_random.random(3), expected)
    assert nested.np_random.bit_generator.state == nested_state_before_restore


def test_nested_wrapper_rng_resolution_is_cycle_safe(recovery_state) -> None:
    env = _FakeEnv()
    del env.wrapper_rng
    first = _EnvLink()
    second = _EnvLink()
    first.env = second
    second.env = first
    second.np_random = np.random.default_rng(505)
    env.env = first

    snapshot = recovery_state.capture_recovery_state(
        env,
        required_capabilities={"wrapper_rng"},
    )
    expected = second.np_random.random(2)
    second.np_random.random(6)
    _restore_bound(recovery_state, env, snapshot)

    np.testing.assert_array_equal(second.np_random.random(2), expected)


def test_snapshot_schema_is_versioned_and_task_bound(recovery_state) -> None:
    env = _FakeEnv()

    snapshot = recovery_state.capture_recovery_state(env)

    assert snapshot.schema_version == recovery_state.RECOVERY_STATE_SCHEMA_VERSION == 3
    assert snapshot.task_identity == recovery_state.PP_BOX_TASK_IDENTITY
    assert snapshot.capabilities.schema_version == 3


def test_capability_mapping_is_structurally_immutable(recovery_state) -> None:
    snapshot = recovery_state.capture_recovery_state(_FakeEnv())

    with pytest.raises(TypeError):
        snapshot.capabilities.available["scene_state"] = False


def test_restore_rejects_unknown_capability_schema_before_mutation(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    incompatible = replace(
        snapshot,
        capabilities=replace(snapshot.capabilities, schema_version=999),
    )
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(
        recovery_state.RecoveryStateSchemaError, match="capability schema"
    ):
        recovery_state.restore_recovery_state(
            env,
            incompatible,
            snapshot_digest="0" * 64,
        )

    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_capture_deep_clones_tensor_and_numpy_scene_state(recovery_state) -> None:
    env = _FakeEnv()

    snapshot = recovery_state.capture_recovery_state(env)
    env.joint_pos.fill_(9.0)
    env.box_root_state.fill(9.0)
    env.recovery_task_state["hands_materialized"].fill_(False)

    torch.testing.assert_close(
        snapshot.scene_state["robot"]["joint_pos"],
        torch.tensor([[0.1, 0.2]], dtype=torch.float32),
    )
    np.testing.assert_array_equal(
        snapshot.scene_state["box"]["root_state"],
        np.arange(13, dtype=np.float32).reshape(1, 13),
    )
    torch.testing.assert_close(
        snapshot.task_state["hands_materialized"],
        torch.tensor([[True, False]]),
    )


def test_global_python_numpy_and_torch_rng_roundtrip(recovery_state) -> None:
    env = _FakeEnv()
    random.seed(10)
    np.random.seed(20)
    torch.manual_seed(30)
    snapshot = recovery_state.capture_recovery_state(env)

    expected = (random.random(), np.random.random(), torch.rand(3))
    _restore_bound(recovery_state, env, snapshot)
    actual = (random.random(), np.random.random(), torch.rand(3))

    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_required_unavailable_capabilities_fail_with_typed_diagnostics(
    recovery_state,
) -> None:
    env = _FakeEnv()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.capture_recovery_state(
            env,
            required_capabilities={"physics_solver_cache", "contact_cache"},
        )

    assert exc_info.value.missing_capabilities == (
        "contact_cache",
        "physics_solver_cache",
    )
    assert "contact_cache" in str(exc_info.value)
    assert "physics_solver_cache" in str(exc_info.value)


def test_capture_only_task_state_hook_is_not_a_capability(recovery_state) -> None:
    env = _CaptureOnlyTaskStateEnv()

    capabilities = recovery_state.discover_recovery_state_capabilities(env)

    assert capabilities.available["task_state"] is False
    assert capabilities.task_state_mode is None
    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.capture_recovery_state(env, required_capabilities={"task_state"})
    assert exc_info.value.missing_capabilities == ("task_state",)


def test_restore_never_creates_fallback_task_state_attribute(recovery_state) -> None:
    source = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(source)
    target = _CaptureOnlyTaskStateEnv()
    env_before = _env_state(target)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        _restore_bound(recovery_state, target, snapshot)

    assert exc_info.value.missing_capabilities == ("task_state",)
    assert not hasattr(target, "recovery_task_state")
    _assert_env_state_equal(_env_state(target), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_direct_task_state_attribute_uses_symmetric_direct_mode(recovery_state) -> None:
    env = _DirectTaskStateEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env.recovery_task_state = {
        "task_success_enabled": False,
        "debug_score": -1,
        "hands_materialized": torch.tensor([[False, False]]),
    }

    _restore_bound(recovery_state, env, snapshot)

    assert snapshot.capabilities.task_state_mode == "direct"
    assert env.recovery_task_state["task_success_enabled"] is True
    assert env.recovery_task_state["debug_score"] == 5
    torch.testing.assert_close(
        env.recovery_task_state["hands_materialized"],
        torch.tensor([[True, False]]),
    )


def test_snapshot_without_task_state_can_restore_into_richer_runtime(
    recovery_state,
) -> None:
    snapshot = recovery_state.capture_recovery_state(_CaptureOnlyTaskStateEnv())
    target = _FakeEnv()

    _restore_bound(recovery_state, target, snapshot)

    assert target.restore_events == ["scene"]
    assert target.recovery_task_state["task_success_enabled"] is True


@pytest.mark.parametrize(
    "payload_name",
    [
        "scene_state",
        "scene_key:robot",
        "scene_key:box",
        "scene_shape",
        "scene_nonfinite",
        "task_counters",
        "counter_shape",
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
        "task_local_rng",
        "wrapper_rng",
        "task_state",
        "task_state_shape",
        "episode_length_buf",
        "common_step_counter",
        "reset_buf",
        "episode_count",
        "reset_count",
    ],
)
def test_incomplete_mandatory_payload_is_rejected_before_any_mutation(
    recovery_state,
    payload_name: str,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    if payload_name == "scene_state":
        incomplete = replace(snapshot, scene_state={})
        expected_missing = "scene_state"
    elif payload_name.startswith("scene_key:"):
        scene_key = payload_name.split(":", maxsplit=1)[1]
        scene_state = dict(snapshot.scene_state)
        scene_state.pop(scene_key)
        incomplete = replace(snapshot, scene_state=scene_state)
        expected_missing = payload_name
    elif payload_name == "scene_shape":
        scene_state = dict(snapshot.scene_state)
        scene_state["robot"] = {"joint_pos": torch.zeros(3, dtype=torch.float32)}
        incomplete = replace(snapshot, scene_state=scene_state)
        expected_missing = "scene_state"
    elif payload_name == "scene_nonfinite":
        scene_state = dict(snapshot.scene_state)
        box_state = snapshot.scene_state["box"]["root_state"].copy()
        box_state[0, 0] = np.nan
        scene_state["box"] = {"root_state": box_state}
        incomplete = replace(snapshot, scene_state=scene_state)
        expected_missing = "scene_state"
    elif payload_name == "task_counters":
        incomplete = replace(snapshot, task_counters=None)
        expected_missing = "task_counters"
    elif payload_name == "counter_shape":
        counters = dict(snapshot.task_counters)
        counters["episode_length_buf"] = torch.tensor([7, 8], dtype=torch.int64)
        incomplete = replace(snapshot, task_counters=counters)
        expected_missing = "task_counter:episode_length_buf"
    elif payload_name == "python_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(snapshot.rng_state, python=("invalid",)),
        )
        expected_missing = "python_rng"
    elif payload_name == "numpy_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(snapshot.rng_state, numpy=("invalid",)),
        )
        expected_missing = "numpy_rng"
    elif payload_name == "torch_cpu_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(
                snapshot.rng_state,
                torch_cpu=torch.empty(0, dtype=torch.uint8),
            ),
        )
        expected_missing = "torch_cpu_rng"
    elif payload_name == "task_local_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(
                snapshot.rng_state,
                task_local=replace(
                    snapshot.rng_state.task_local,
                    state=torch.empty(0, dtype=torch.uint8),
                ),
            ),
        )
        expected_missing = "task_local_rng"
    elif payload_name == "wrapper_rng":
        incomplete = replace(
            snapshot,
            rng_state=replace(
                snapshot.rng_state,
                wrapper=replace(snapshot.rng_state.wrapper, state={"invalid": True}),
            ),
        )
        expected_missing = "wrapper_rng"
    elif payload_name == "task_state":
        incomplete = replace(snapshot, task_state=None)
        expected_missing = "task_state"
    elif payload_name == "task_state_shape":
        task_state = dict(snapshot.task_state)
        task_state["hands_materialized"] = torch.tensor([True, False])
        incomplete = replace(snapshot, task_state=task_state)
        expected_missing = "task_state"
    else:
        counters = dict(snapshot.task_counters)
        counters.pop(payload_name)
        incomplete = replace(snapshot, task_counters=counters)
        expected_missing = f"task_counter:{payload_name}"
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        _restore_bound(recovery_state, env, incomplete)

    assert expected_missing in exc_info.value.missing_capabilities
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_missing_rng_record_reports_every_advertised_rng_before_mutation(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    incomplete = replace(snapshot, rng_state=None)
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        _restore_bound(recovery_state, env, incomplete)

    assert set(exc_info.value.missing_capabilities) >= {
        "python_rng",
        "numpy_rng",
        "torch_cpu_rng",
        "task_local_rng",
        "wrapper_rng",
    }
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_restore_rejects_missing_required_cache_payload_before_reset(
    recovery_state,
) -> None:
    env = _CacheEnv()
    snapshot = recovery_state.capture_recovery_state(
        env,
        required_capabilities={"physics_solver_cache"},
    )
    incomplete = replace(snapshot, runtime_state={})

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        _restore_bound(
            recovery_state,
            env,
            incomplete,
            required_capabilities={"physics_solver_cache"},
        )

    assert exc_info.value.missing_capabilities == ("physics_solver_cache",)
    assert env.restore_events == []


@pytest.mark.parametrize(
    "runtime_state",
    [None, ["physics_solver_cache", "contact_cache"]],
)
def test_restore_rejects_non_mapping_runtime_cache_state_before_any_mutation(
    recovery_state,
    runtime_state: object,
) -> None:
    env = _BothCacheEnv()
    snapshot = recovery_state.capture_recovery_state(
        env,
        required_capabilities={"physics_solver_cache", "contact_cache"},
    )
    incomplete = replace(snapshot, runtime_state=runtime_state)
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        _restore_bound(recovery_state, env, incomplete)

    assert exc_info.value.missing_capabilities == (
        "contact_cache",
        "physics_solver_cache",
    )
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_capture_records_scene_task_counters_and_optional_task_state(
    recovery_state,
) -> None:
    env = _FakeEnv()

    snapshot = recovery_state.capture_recovery_state(env)

    assert snapshot.capabilities.available["scene_state"] is True
    assert snapshot.capabilities.available["task_counters"] is True
    assert snapshot.capabilities.available["task_state"] is True
    assert snapshot.capabilities.available["task_local_rng"] is True
    assert snapshot.capabilities.available["wrapper_rng"] is True
    assert snapshot.capabilities.available["physics_solver_cache"] is False
    assert snapshot.capabilities.available["contact_cache"] is False
    assert snapshot.capabilities.task_state_mode == "hooks"
    assert snapshot.capabilities.scene_state_keys == ("robot", "box")
    assert snapshot.capabilities.counter_names == (
        "episode_length_buf",
        "common_step_counter",
        "reset_buf",
        "episode_count",
        "reset_count",
    )
    torch.testing.assert_close(
        snapshot.task_counters["episode_length_buf"], torch.tensor([7])
    )
    torch.testing.assert_close(
        snapshot.task_counters["reset_buf"], torch.tensor([False])
    )
    assert snapshot.task_counters["common_step_counter"] == 11
    assert snapshot.task_counters["episode_count"] == 3
    torch.testing.assert_close(snapshot.task_counters["reset_count"], torch.tensor([2]))


def test_exact_continuation_fails_closed_without_control_and_cache_participants(
    recovery_state,
) -> None:
    env = _FakeEnv()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.capture_recovery_state(
            env,
            fidelity_tier="exact_continuation",
        )

    assert set(exc_info.value.missing_capabilities) >= {
        "action_manager_state",
        "action_provider_state",
        "controller_state",
        "physics_solver_cache",
        "contact_cache",
    }


def test_exact_continuation_snapshot_binds_all_control_participants(
    recovery_state,
) -> None:
    env = _ExactControlEnv()

    snapshot = recovery_state.capture_recovery_state(
        env,
        fidelity_tier="exact_continuation",
    )

    assert snapshot.fidelity_tier == "exact_continuation"
    assert set(snapshot.runtime_state) >= {
        "action_manager_state",
        "action_provider_state",
        "controller_state",
        "physics_solver_cache",
        "contact_cache",
    }
    env.action_provider.value = 41
    env.recovery_controller.value = 42
    env.action_manager.value = 43
    _restore_bound(recovery_state, env, snapshot)
    assert env.action_provider.value == 31
    assert env.recovery_controller.value == 32
    assert env.action_manager.value == 33


@pytest.mark.parametrize("participant_name", ["action_provider", "recovery_controller"])
def test_late_control_participant_failure_rolls_back_all_control_state(
    recovery_state,
    participant_name: str,
) -> None:
    env = _ExactControlEnv()
    target = recovery_state.capture_recovery_state(
        env,
        fidelity_tier="exact_continuation",
    )
    env.deterministic_step()
    env.action_provider.value = 41
    env.recovery_controller.value = 42
    env.action_manager.value = 43
    rollback_values = (41, 42, 43)
    rollback = recovery_state.capture_recovery_state(
        env,
        fidelity_tier="exact_continuation",
    )
    rollback_digest = recovery_state.recovery_state_digest(rollback)
    getattr(env, participant_name).restore_failures_remaining = 1

    with pytest.raises(recovery_state.RecoveryStateTransactionError) as exc_info:
        _restore_bound(recovery_state, env, target)

    assert exc_info.value.evidence.rollback_succeeded is True
    assert (
        env.action_provider.value,
        env.recovery_controller.value,
        env.action_manager.value,
    ) == rollback_values
    restored = recovery_state.capture_recovery_state(
        env,
        fidelity_tier="exact_continuation",
    )
    assert recovery_state.recovery_state_digest(restored) == rollback_digest


def test_control_participant_rollback_failure_preserves_both_errors(
    recovery_state,
) -> None:
    env = _ExactControlEnv()
    target = recovery_state.capture_recovery_state(
        env,
        fidelity_tier="exact_continuation",
    )
    env.action_provider.value = 41
    env.action_provider.restore_failures_remaining = 2

    with pytest.raises(recovery_state.RecoveryStateTransactionError) as exc_info:
        _restore_bound(recovery_state, env, target)

    evidence = exc_info.value.evidence
    assert evidence.failure_type == "RuntimeError"
    assert "action_provider_state" in evidence.failure_message
    assert evidence.rollback_succeeded is False
    assert evidence.rollback_failure_type == "RuntimeError"
    assert "action_provider_state" in evidence.rollback_failure_message


def test_restore_uses_scene_reset_then_counters_task_state_and_rng(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env.episode_length_buf.fill_(99)
    env.common_step_counter = 99
    env.reset_buf.fill_(True)
    env.episode_count = 99
    env.reset_count.fill_(99)
    env.recovery_task_state = {"task_success_enabled": False}

    _restore_bound(recovery_state, env, snapshot)

    assert env.restore_events == ["scene", "task"]
    torch.testing.assert_close(env.episode_length_buf, torch.tensor([7]))
    assert env.common_step_counter == 11
    torch.testing.assert_close(env.reset_buf, torch.tensor([False]))
    assert env.episode_count == 3
    torch.testing.assert_close(env.reset_count, torch.tensor([2]))
    assert env.recovery_task_state["task_success_enabled"] is True


def test_restore_rejects_snapshot_from_unknown_schema(recovery_state) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    incompatible = recovery_state.RecoveryStateSnapshot(
        schema_version=999,
        task_identity=snapshot.task_identity,
        scope=snapshot.scope,
        fidelity_tier=snapshot.fidelity_tier,
        capabilities=snapshot.capabilities,
        scene_state=snapshot.scene_state,
        task_counters=snapshot.task_counters,
        task_state=snapshot.task_state,
        rng_state=snapshot.rng_state,
    )

    with pytest.raises(recovery_state.RecoveryStateSchemaError, match="schema version"):
        recovery_state.restore_recovery_state(
            env,
            incompatible,
            snapshot_digest="0" * 64,
        )


def test_restore_replays_the_same_next_observable_step(recovery_state) -> None:
    env = _FakeEnv()
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    snapshot = recovery_state.capture_recovery_state(env)

    expected = env.deterministic_step()
    env.deterministic_step()
    env.joint_pos.fill_(-10.0)
    env.box_root_state.fill(-20.0)
    env.episode_length_buf.fill_(90)
    env.common_step_counter = 91
    env.reset_buf.fill_(True)
    env.episode_count = 92
    env.reset_count.fill_(93)
    env.recovery_task_state = {
        "task_success_enabled": False,
        "debug_score": -94,
        "hands_materialized": torch.tensor([[False, False]]),
    }
    _restore_bound(recovery_state, env, snapshot)
    actual = env.deterministic_step()

    _assert_transition_equal(actual, expected)


def test_restore_requires_caller_bound_snapshot_digest(recovery_state) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)

    with pytest.raises(TypeError, match="snapshot_digest"):
        recovery_state.restore_recovery_state(env, snapshot)


def test_restore_rejects_digest_mismatch_before_environment_access(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env.api_events.clear()
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateDigestMismatchError):
        recovery_state.restore_recovery_state(
            env,
            snapshot,
            snapshot_digest="0" * 64,
        )

    assert env.api_events == []
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_task_state_preflight_failure_happens_before_reset_or_rollback_capture(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env.api_events.clear()
    env.preflight_failure = ValueError("task state is not restorable")
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStatePreflightError, match="task_state"):
        recovery_state.restore_recovery_state(
            env,
            snapshot,
            snapshot_digest=recovery_state.recovery_state_digest(snapshot),
        )

    assert env.api_events == [("get_state", True)]
    assert env.restore_events == []
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_read_only_task_counter_is_rejected_before_reset(recovery_state) -> None:
    env = _ReadOnlyCounterEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env.api_events.clear()
    env_before = _env_state(env)
    rng_before = _global_rng_state()

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        _restore_bound(recovery_state, env, snapshot)

    assert "task_counter:common_step_counter" in exc_info.value.missing_capabilities
    assert env.api_events == [("get_state", True)]
    assert env.restore_events == []
    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_failed_restore_rolls_back_exact_state_and_emits_success_evidence(
    recovery_state,
) -> None:
    env = _FakeEnv()
    target = recovery_state.capture_recovery_state(env)
    env.deterministic_step()
    rollback = recovery_state.capture_recovery_state(env)
    rollback_digest = recovery_state.recovery_state_digest(rollback)
    env.restore_failures_remaining = 1

    with pytest.raises(recovery_state.RecoveryStateTransactionError) as exc_info:
        recovery_state.restore_recovery_state(
            env,
            target,
            snapshot_digest=recovery_state.recovery_state_digest(target),
        )

    evidence = exc_info.value.evidence
    assert evidence.failure_phase == "apply_target"
    assert evidence.failure_type == "RuntimeError"
    assert evidence.rollback_succeeded is True
    assert evidence.rollback_failure_type is None
    assert evidence.rollback_snapshot_digest == rollback_digest
    restored = recovery_state.capture_recovery_state(env)
    assert recovery_state.recovery_state_digest(restored) == rollback_digest


def test_postcondition_digest_failure_rolls_back_with_verify_phase_evidence(
    recovery_state,
) -> None:
    env = _FakeEnv()
    target = recovery_state.capture_recovery_state(env)
    env.deterministic_step()
    rollback = recovery_state.capture_recovery_state(env)
    rollback_digest = recovery_state.recovery_state_digest(rollback)
    env.corrupt_restores_remaining = 1

    with pytest.raises(recovery_state.RecoveryStateTransactionError) as exc_info:
        _restore_bound(recovery_state, env, target)

    evidence = exc_info.value.evidence
    assert evidence.failure_phase == "verify_target"
    assert evidence.failure_type == "RecoveryStateDigestMismatchError"
    assert evidence.rollback_succeeded is True
    restored = recovery_state.capture_recovery_state(env)
    assert recovery_state.recovery_state_digest(restored) == rollback_digest


def test_failed_rollback_emits_both_target_and_rollback_failure_evidence(
    recovery_state,
) -> None:
    env = _FakeEnv()
    target = recovery_state.capture_recovery_state(env)
    env.deterministic_step()
    env.restore_failures_remaining = 2

    with pytest.raises(recovery_state.RecoveryStateTransactionError) as exc_info:
        recovery_state.restore_recovery_state(
            env,
            target,
            snapshot_digest=recovery_state.recovery_state_digest(target),
        )

    evidence = exc_info.value.evidence
    assert evidence.failure_type == "RuntimeError"
    assert evidence.rollback_succeeded is False
    assert evidence.rollback_failure_type == "RuntimeError"
    assert "injected task-state restore failure" in evidence.rollback_failure_message
    assert len(evidence.target_snapshot_digest) == 64
    assert len(evidence.rollback_snapshot_digest) == 64


def test_restore_fails_closed_when_process_global_rng_exclusion_is_busy(
    recovery_state,
) -> None:
    env = _FakeEnv()
    snapshot = recovery_state.capture_recovery_state(env)
    env_before = _env_state(env)
    rng_before = _global_rng_state()
    assert recovery_state._PROCESS_GLOBAL_RNG_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(recovery_state.RecoveryStateGlobalRngBusyError):
            recovery_state.restore_recovery_state(
                env,
                snapshot,
                snapshot_digest=recovery_state.recovery_state_digest(snapshot),
            )
    finally:
        recovery_state._PROCESS_GLOBAL_RNG_LOCK.release()

    _assert_env_state_equal(_env_state(env), env_before)
    _assert_global_rng_state_equal(_global_rng_state(), rng_before)


def test_pp_box_production_hooks_restore_every_reset_to_mutated_buffer(
    recovery_state,
) -> None:
    env = _ProductionEnv(recovery_state.PP_BOX_TASK_IDENTITY)
    recovery_state.install_pp_box_recovery_task_state_hooks(env)
    snapshot = recovery_state.capture_recovery_state(
        env,
        required_capabilities={
            "task_state",
            "process_global_rng_exclusive",
            "action_manager_state",
        },
    )

    assert snapshot.capabilities.available["action_manager_state"] is True
    assert "action_manager_state" in snapshot.runtime_state

    recovery_state.restore_recovery_state(
        env,
        snapshot,
        snapshot_digest=recovery_state.recovery_state_digest(snapshot),
    )

    assert env.reset_calls == 1
    torch.testing.assert_close(env.action_manager._action, torch.tensor([[1.0, 2.0]]))
    torch.testing.assert_close(
        env.action_manager._prev_action, torch.tensor([[3.0, 4.0]])
    )
    term = env.action_manager._terms["joint_pos"]
    torch.testing.assert_close(term._raw_actions, torch.tensor([[5.0, 6.0]]))
    torch.testing.assert_close(term._processed_actions, torch.tensor([[7.0, 8.0]]))
    torch.testing.assert_close(
        env.reward_manager._episode_sums["reward"], torch.tensor([9.0])
    )
    torch.testing.assert_close(env.reward_manager._reward_buf, torch.tensor([10.0]))
    torch.testing.assert_close(env.reward_manager._step_reward, torch.tensor([[11.0]]))
    torch.testing.assert_close(
        env.termination_manager._truncated_buf, torch.tensor([False])
    )
    torch.testing.assert_close(
        env.termination_manager._terminated_buf, torch.tensor([True])
    )
    torch.testing.assert_close(
        env.observation_manager._obs_buffer["policy"]["robot_joint_state"],
        torch.tensor([[12.0, 13.0]]),
    )
    assert env.obs_buf is env.observation_manager._obs_buffer
    assert env.reward_buf is env.reward_manager._reward_buf
    assert env.reset_terminated is env.termination_manager._terminated_buf
    assert env.reset_time_outs is env.termination_manager._truncated_buf
    assert env._sim_step_counter == 14
    torch.testing.assert_close(env.extras["log"]["episode"], torch.tensor([15.0]))
    torch.testing.assert_close(
        env.scene.robot.data.joint_pos_target,
        torch.tensor([[16.0, 17.0]]),
    )
    torch.testing.assert_close(
        env.scene.robot.data.joint_vel_target,
        torch.tensor([[18.0, 19.0]]),
    )
    torch.testing.assert_close(
        env.scene.robot.data.joint_effort_target,
        torch.tensor([[20.0, 21.0]]),
    )
    assert env.cfg._episode_runtime_seed == 22
    assert env.cfg._episode_object_seed_counter == 23
    assert env.cfg._current_episode_object_seed == 24
    assert env.cfg._current_episode_object_seed_source == "env_seed"
    assert env.cfg._replay_initial_env_state_active is True


def test_pp_box_hook_install_initializes_pre_step_reset_buffer(
    recovery_state,
) -> None:
    env = _ProductionEnv(recovery_state.PP_BOX_TASK_IDENTITY)
    for name in (
        "reset_buf",
        "reward_buf",
        "reset_terminated",
        "reset_time_outs",
        "obs_buf",
    ):
        delattr(env, name)

    recovery_state.install_pp_box_recovery_task_state_hooks(env)

    assert torch.equal(env.reset_buf, torch.tensor([True]))
    assert env.reward_buf is env.reward_manager._reward_buf
    assert env.reset_terminated is env.termination_manager._terminated_buf
    assert env.reset_time_outs is env.termination_manager._truncated_buf
    assert env.obs_buf is env.observation_manager._obs_buffer
    snapshot = recovery_state.capture_recovery_state(env, fidelity_tier="state_only")
    assert snapshot.capabilities.available["task_counters"] is True
    assert torch.equal(snapshot.task_counters["reset_buf"], torch.tensor([True]))


def test_pp_box_task_state_key_mismatch_preserves_both_key_sets(
    recovery_state,
) -> None:
    with pytest.raises(
        recovery_state.RecoveryStateSchemaError,
        match=r"saved_keys=\('stale',\), runtime_keys=\(\)",
    ):
        recovery_state._require_exact_mapping_keys(
            {"stale": torch.tensor([False])},
            set(),
            path="termination_manager.term_dones",
        )


def test_pp_box_hook_install_preflights_all_participants_before_binding(
    recovery_state,
) -> None:
    env = _ProductionEnv(recovery_state.PP_BOX_TASK_IDENTITY)
    env.action_manager.capture_recovery_action_manager_state = dict

    with pytest.raises(
        recovery_state.RecoveryStateIncompleteError,
        match="conflicting_hook:action_manager",
    ):
        recovery_state.install_pp_box_recovery_task_state_hooks(env)

    assert not hasattr(env, "capture_recovery_task_state")
    assert not hasattr(env, "preflight_restore_recovery_task_state")
    assert not hasattr(env, "restore_recovery_task_state")
    assert not hasattr(env, "_pp_box_recovery_task_state_hooks_version")


def test_pp_box_installs_bound_recovery_state_coordinator(recovery_state) -> None:
    env = _ProductionEnv(recovery_state.PP_BOX_TASK_IDENTITY)
    recovery_state.install_pp_box_recovery_task_state_hooks(env)
    coordinator = env.recovery_state_coordinator

    assert type(coordinator) is recovery_state.RecoveryStateCoordinator
    assert coordinator.binding_identity == {
        "schema_version": 1,
        "coordinator_type": ("pp_box_recovery_state.RecoveryStateCoordinator"),
        "task_identity": recovery_state.PP_BOX_TASK_IDENTITY,
    }
    snapshot = coordinator.capture(fidelity_tier="state_only")
    snapshot_digest = coordinator.digest(snapshot)
    coordinator.preflight(snapshot, snapshot_digest=snapshot_digest)
    env.joint_pos.fill_(-5.0)
    coordinator.restore(snapshot, snapshot_digest=snapshot_digest)
    torch.testing.assert_close(env.joint_pos, torch.tensor([[0.1, 0.2]]))

    env.recovery_state_coordinator = object()
    with pytest.raises(
        recovery_state.RecoveryStateSchemaError,
        match="coordinator binding",
    ):
        coordinator.capture(fidelity_tier="state_only")


def test_pp_box_production_hook_accepts_rlinf_observation_contract(
    recovery_state,
) -> None:
    env = _ProductionEnv(recovery_state.PP_BOX_TASK_IDENTITY)
    env.observation_manager._group_obs_term_names["policy"].extend(
        [
            "vla_state64",
            "vla_front_rgb",
            "up_alignment",
            "critical_contact_force_max",
            "critical_contact_force_available",
        ]
    )
    recovery_state.install_pp_box_recovery_task_state_hooks(env)

    snapshot = recovery_state.capture_recovery_state(
        env,
        required_capabilities={"task_state", "process_global_rng_exclusive"},
    )

    assert snapshot.task_state["runtime_identity"]["observation_terms"] == {
        "policy": (
            "robot_joint_state",
            "robot_gipper_state",
            "camera_image",
            "vla_state64",
            "vla_front_rgb",
            "up_alignment",
            "critical_contact_force_max",
            "critical_contact_force_available",
        )
    }


def test_pp_box_production_hook_rejects_unknown_observation_extension(
    recovery_state,
) -> None:
    env = _ProductionEnv(recovery_state.PP_BOX_TASK_IDENTITY)
    env.observation_manager._group_obs_term_names["policy"].append("unknown_term")
    recovery_state.install_pp_box_recovery_task_state_hooks(env)

    with pytest.raises(
        recovery_state.RecoveryStateIncompleteError,
        match="observation_manager.term_contract",
    ):
        recovery_state.capture_recovery_state(env)


def test_pp_box_production_hook_rejects_wrong_task_identity(recovery_state) -> None:
    env = _ProductionEnv("Isaac-Another-Task")

    with pytest.raises(recovery_state.RecoveryStateSchemaError, match="cfg.env_name"):
        recovery_state.install_pp_box_recovery_task_state_hooks(env)


def test_pp_box_cfg_wires_task_identity_and_production_state_hooks() -> None:
    tree = ast.parse(PP_BOX_CFG_PATH.read_text(encoding="utf-8"))
    cfg_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "MovePickPlaceBoxG129Dex3WholedobyEnvCfg"
    )
    identity_assignment = next(
        node
        for node in cfg_class.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "recovery_task_identity"
            for target in node.targets
        )
    )
    assert isinstance(identity_assignment.value, ast.Attribute)
    assert identity_assignment.value.attr == "PP_BOX_TASK_IDENTITY"
    initializer = next(
        node
        for node in cfg_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "initialize_task_scene"
    )
    calls = [node for node in ast.walk(initializer) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "install_pp_box_recovery_task_state_hooks"
        and len(call.args) == 1
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "env"
        for call in calls
    )


def _real_sonic_provider_for_recovery_test():
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    from action_provider.action_provider_sonic import SonicActionProvider

    provider = object.__new__(SonicActionProvider)
    provider._use_lerobot_vla = True
    provider.name = "SonicActionProvider"
    provider._use_vla_latent64 = False
    provider._vla_action_format = "semantic_v3"
    provider._replay_enabled = False
    provider._sonic_output_delay_steps = 0
    provider._lerobot_action_chunk_queue = deque(
        [
            np.linspace(0.0, 0.39, 40, dtype=np.float32),
            np.linspace(1.0, 1.39, 40, dtype=np.float32),
        ]
    )
    array_shapes = {
        "_smpl_joints_buf": (10, 24, 3),
        "_smpl_pose_buf": (10, 21, 3),
        "_body_rot6d_buf": (10, 6),
        "_ref_smpl_joints_window": (10, 24, 3),
        "_ref_body_quat_window": (10, 4),
        "_ref_joint_pos_window": (10, 29),
        "_robot_joint_pos_hist": (10, 29),
        "_robot_joint_vel_hist": (10, 29),
        "_motion_joint_pos_hist": (50, 29),
        "_motion_joint_vel_hist": (50, 29),
        "_motion_root_z_hist": (50,),
        "_motion_anchor_rot6d_hist": (50, 6),
        "_ang_vel_hist": (10, 3),
        "_grav_dir_hist": (10, 3),
        "_last_action_hist": (10, 29),
        "_left_hand_target": (7,),
        "_right_hand_target": (7,),
        "_vr_3pt_position": (9,),
        "_vr_3pt_orientation": (12,),
        "_anchor_init_base_quat_wxyz": (4,),
        "_anchor_init_ref_quat_wxyz": (4,),
        "_anchor_heading_align_quat_wxyz": (4,),
        "_tracking_target_buffer": (29,),
        "_latest_canonical_action_raw": (40,),
        "_latest_canonical_action": (40,),
        "_sonic_last_executed_target": (29,),
        "_latest_executed_canonical_action_raw": (40,),
        "_latest_executed_canonical_action": (40,),
        "_latest_encoder_input": (1762,),
        "_latest_smpl_joint_window": (10, 24, 3),
        "_latest_anchor_window": (10, 6),
        "_latest_wrist_window": (10, 6),
        "_latest_decoder_obs": (994,),
        "_latest_decoder_raw_action": (29,),
        "_latest_decoder_target": (29,),
        "_latest_decoder_body_effort": (29,),
        "_latest_aligned_body_quat_wxyz": (4,),
        "_latest_consumed_anchor_rot6d": (6,),
    }
    for index, (name, shape) in enumerate(array_shapes.items(), start=1):
        setattr(provider, name, np.full(shape, index / 100.0, dtype=np.float32))
    provider._vla_initial_robot_quat_wxyz = np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    provider._vla_prev_root_rot6d_action = np.arange(6, dtype=np.float32)
    provider._latest_vla_action = np.arange(40, dtype=np.float32)
    provider._ref_window_valid = True
    provider._left_hand_binary_state = False
    provider._right_hand_binary_state = True
    provider._vla_semantic_history_fill = 7
    provider._smpl_data_valid = True
    provider._frame_count = 18
    provider._smpl_history_fill = 10
    provider._anchor_heading_initialized = True
    provider._anchor_use_heading_align = True
    provider._stream_ref_frames = {17: {"joint_pos": np.ones((29,), dtype=np.float32)}}
    provider._stream_ref_indices = [17]
    provider._stream_playback_frame_idx = 17
    provider._stream_window_start = 17
    provider._stream_current_frame = 0
    provider._stream_frame_step = 1
    provider._latest_frame_index = 23
    provider._latest_timestamp_realtime = 1.25
    provider._latest_timestamp_monotonic = 2.5
    provider._latest_heading_increment = 0.125
    provider._latest_consumed_new_this_step = True
    provider._effort_mode_runtime_configured = False
    provider._position_mode_runtime_configured = True
    provider._latest_consumed_control_step = 24
    provider._raw_input_frame_index = 23
    provider._last_raw_frame_index = 23
    provider._raw_input_timestamp_realtime = 1.25
    provider._raw_input_timestamp_monotonic = 2.5
    provider._raw_pose_payload = {"joint_pos": np.arange(29, dtype=np.float32)}
    provider._latest_pose_payload = {"joint_pos": np.arange(29, dtype=np.float32)}
    provider._latest_human_smplx_frame = None
    provider._latent = np.arange(64, dtype=np.float32).reshape(1, 64)
    provider._sonic_output_delay_queue = []
    provider._sonic_last_executed_bundle = {
        "body_action_29dof": np.arange(29, dtype=np.float32),
        "canonical_action_raw": np.arange(40, dtype=np.float32),
        "canonical_action_aligned": np.arange(40, dtype=np.float32),
        "source_frame_index": 23,
        "source_timestamp_realtime": 1.25,
        "source_timestamp_monotonic": 2.5,
        "source_control_step": 24,
    }
    provider._latest_executed_source_frame_index = 23
    provider._latest_executed_source_timestamp_realtime = 1.25
    provider._latest_executed_source_timestamp_monotonic = 2.5
    provider._latest_executed_source_control_step = 24
    provider._lerobot_vla_runtime = SimpleNamespace(
        _body_xy_world=np.array([0.1, 0.2], dtype=np.float32),
        _body_z_world=0.8,
        _prev_target_root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        _episode_ref_to_world_heading_quat_wxyz=np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        ),
        _prev_root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        _prev_action_rel_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        _prev_joint_pos_canonical_29=np.arange(29, dtype=np.float32),
        _last_selected_root_rot6d_layout="row",
    )
    return provider


def test_real_sonic_provider_recovery_hooks_roundtrip_queue_history_and_execution() -> (
    None
):
    provider = _real_sonic_provider_for_recovery_test()

    snapshot = provider.capture_recovery_provider_state()
    expected_first_row = snapshot["committed_action_queue"][0].copy()
    expected_hist = snapshot["fields"]["_last_action_hist"].copy()
    expected_runtime_xy = snapshot["vla_runtime"]["_body_xy_world"].copy()

    provider._lerobot_action_chunk_queue.popleft()
    provider._last_action_hist.fill(-9.0)
    provider._stream_playback_frame_idx = 99
    provider._sonic_last_executed_target.fill(-8.0)
    provider._latest_executed_source_control_step = 99
    provider._lerobot_vla_runtime._body_xy_world.fill(-7.0)
    provider.restore_recovery_provider_state(snapshot)

    assert len(provider._lerobot_action_chunk_queue) == 2
    np.testing.assert_array_equal(
        provider._lerobot_action_chunk_queue[0], expected_first_row
    )
    np.testing.assert_array_equal(provider._last_action_hist, expected_hist)
    assert provider._stream_playback_frame_idx == 17
    np.testing.assert_array_equal(
        provider._sonic_last_executed_target,
        snapshot["fields"]["_sonic_last_executed_target"],
    )
    assert provider._latest_executed_source_control_step == 24
    np.testing.assert_array_equal(
        provider._lerobot_vla_runtime._body_xy_world, expected_runtime_xy
    )


def test_real_sonic_provider_recovery_preflight_rejects_bad_committed_row() -> None:
    provider = _real_sonic_provider_for_recovery_test()
    snapshot = provider.capture_recovery_provider_state()
    snapshot["committed_action_queue"] = (np.zeros((39,), dtype=np.float32),)

    with pytest.raises(ValueError, match="committed_action_queue"):
        provider.preflight_restore_recovery_provider_state(snapshot)


def test_real_sonic_provider_capture_rejects_non_float32_live_committed_row() -> None:
    provider = _real_sonic_provider_for_recovery_test()
    provider._lerobot_action_chunk_queue[0] = np.zeros((40,), dtype=np.float64)

    with pytest.raises(ValueError, match="committed_action_queue"):
        provider.capture_recovery_provider_state()


def test_real_robot_controller_recovery_hooks_roundtrip_and_require_env_aliases() -> (
    None
):
    from layeredcontrol.robot_control_system import ControlConfig, RobotController

    provider = _real_sonic_provider_for_recovery_test()
    controller = object.__new__(RobotController)
    controller.config = ControlConfig(
        step_hz=50,
        replay_mode=False,
        use_rl_action_mode=True,
    )
    controller._last_action = torch.arange(43, dtype=torch.float32)
    controller.step_count = 77
    controller.action_provider = None
    controller.env = SimpleNamespace()
    controller.set_action_provider(provider)

    assert controller.env.action_provider is provider
    assert controller.env.recovery_controller is controller

    snapshot = controller.capture_recovery_controller_state()
    controller._last_action.fill_(-1.0)
    controller.step_count = 99
    controller.restore_recovery_controller_state(snapshot)

    torch.testing.assert_close(
        controller._last_action, torch.arange(43, dtype=torch.float32)
    )
    assert controller.step_count == 77
    controller.env.action_provider = object()
    with pytest.raises(ValueError, match="env.action_provider"):
        controller.preflight_restore_recovery_controller_state(snapshot)


@pytest.mark.parametrize("launcher_path", [SIM_MAIN_PATH, SIM_EVAL_VLA_PATH])
def test_production_launchers_attach_provider_and_controller_to_environment(
    launcher_path: Path,
) -> None:
    tree = ast.parse(launcher_path.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "controller"
        and call.func.attr == "set_action_provider"
        for call in calls
    )
    assignments = [node for node in ast.walk(main) if isinstance(node, ast.Assign)]
    assert any(
        any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "env"
            and target.attr == "action_provider"
            for target in assignment.targets
        )
        and isinstance(assignment.value, ast.Attribute)
        and isinstance(assignment.value.value, ast.Name)
        and assignment.value.value.id == "controller"
        and assignment.value.attr == "action_provider"
        for assignment in assignments
    )
    assert any(
        any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "env"
            and target.attr == "recovery_controller"
            for target in assignment.targets
        )
        and isinstance(assignment.value, ast.Name)
        and assignment.value.id == "controller"
        for assignment in assignments
    )


def test_production_shaped_exact_snapshot_roundtrips_real_sonic_and_controller() -> (
    None
):
    recovery_state = _load_recovery_state_module()
    from layeredcontrol.robot_control_system import ControlConfig, RobotController

    env = _ProductionExactEnv(recovery_state.PP_BOX_TASK_IDENTITY)
    recovery_state.install_pp_box_recovery_task_state_hooks(env)
    provider = _real_sonic_provider_for_recovery_test()
    controller = object.__new__(RobotController)
    controller.config = ControlConfig(
        step_hz=50,
        replay_mode=False,
        use_rl_action_mode=True,
    )
    controller._last_action = torch.arange(43, dtype=torch.float32)
    controller.step_count = 77
    controller.action_provider = None
    controller.env = env
    controller.set_action_provider(provider)

    snapshot = recovery_state.capture_recovery_state(
        env,
        fidelity_tier="exact_continuation",
    )
    expected_digest = recovery_state.recovery_state_digest(snapshot)
    env.joint_pos.fill_(-2.0)
    env.action_manager._action.fill_(-3.0)
    provider._lerobot_action_chunk_queue.popleft()
    provider._last_action_hist.fill(-4.0)
    controller._last_action.fill_(-5.0)
    controller.step_count = 99
    env.physics_solver_epoch = 40
    env.contact_epoch = 60

    recovery_state.restore_recovery_state(
        env,
        snapshot,
        snapshot_digest=expected_digest,
    )

    restored = recovery_state.capture_recovery_state(
        env,
        fidelity_tier="exact_continuation",
    )
    assert recovery_state.recovery_state_digest(restored) == expected_digest
    assert len(provider._lerobot_action_chunk_queue) == 2
    torch.testing.assert_close(
        controller._last_action, torch.arange(43, dtype=torch.float32)
    )
    assert controller.step_count == 77


def test_exact_snapshot_reports_empty_committed_sonic_queue_as_typed_unsupported() -> (
    None
):
    recovery_state = _load_recovery_state_module()
    from layeredcontrol.robot_control_system import ControlConfig, RobotController

    env = _ProductionExactEnv(recovery_state.PP_BOX_TASK_IDENTITY)
    recovery_state.install_pp_box_recovery_task_state_hooks(env)
    provider = _real_sonic_provider_for_recovery_test()
    provider._lerobot_action_chunk_queue.clear()
    controller = object.__new__(RobotController)
    controller.config = ControlConfig(use_rl_action_mode=True)
    controller._last_action = torch.zeros(43)
    controller.step_count = 0
    controller.action_provider = None
    controller.env = env
    controller.set_action_provider(provider)

    with pytest.raises(recovery_state.RecoveryStateIncompleteError) as exc_info:
        recovery_state.capture_recovery_state(
            env,
            fidelity_tier="exact_continuation",
        )

    assert "action_provider_state" in exc_info.value.missing_capabilities
